"""Per-model profiles, global templates, and exposing a profile as a model."""

from __future__ import annotations

import json

import pytest

from bwr.server.profiles import (
    ENGINE_FIELDS,
    ProfileError,
    ProfileStore,
    exposed_id,
    split_exposed,
)


@pytest.fixture
def store(tmp_path):
    return ProfileStore(tmp_path)


# -- templates ---------------------------------------------------------------


def test_builtin_templates_are_present_and_read_only(store):
    names = {t.name for t in store.list_templates()}
    assert {"precise", "balanced", "creative"} <= names
    with pytest.raises(ProfileError):
        store.update_template("precise", display_name="Mine")
    with pytest.raises(ProfileError):
        store.delete_template("precise")


def test_builtin_templates_are_sampling_only(store):
    """Every shipped template must be exposable without a reload; one that
    set an engine field would silently be a different kind of thing."""
    for t in store.list_templates():
        if t.is_builtin:
            assert not (set(t.settings) & ENGINE_FIELDS), t.name


def test_a_user_template_round_trips_to_disk(store, tmp_path):
    store.create_template("mine", display_name="Mine", settings={"top_p": 0.5})
    reloaded = ProfileStore(tmp_path)
    got = {t.name: t for t in reloaded.list_templates()}["mine"]
    assert got.settings == {"top_p": 0.5}
    assert got.is_builtin is False


def test_a_builtin_name_on_disk_never_overrides_the_shipped_one(store, tmp_path):
    """A file claiming to define `precise` must not shadow the built-in --
    otherwise editing the file is a way around the read-only rule."""
    path = tmp_path / ".bwr" / "profiles.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "templates": {"precise": {"display_name": "Hijacked", "settings": {"top_p": 0.1}}}
    }))
    reloaded = ProfileStore(tmp_path)
    precise = {t.name: t for t in reloaded.list_templates()}["precise"]
    assert precise.display_name == "Precise"
    assert precise.is_builtin is True


# -- per-model profiles ------------------------------------------------------


def test_create_and_list_a_profile(store):
    store.create_profile("m1", "fast", settings={"temperature": 0.1})
    assert [p.name for p in store.list_profiles("m1")] == ["fast"]
    assert store.list_profiles("m2") == []


def test_duplicate_names_are_rejected(store):
    store.create_profile("m1", "fast")
    with pytest.raises(ProfileError):
        store.create_profile("m1", "fast")


@pytest.mark.parametrize("bad", ["", "has space", "../escape", "-leading", "a" * 65])
def test_invalid_names_are_rejected(store, bad):
    """Names become URL path segments and part of an exposed model id."""
    with pytest.raises(ProfileError):
        store.create_profile("m1", bad)


def test_a_template_seeds_a_profile_but_explicit_settings_win(store):
    store.create_template("base", settings={"temperature": 0.7, "top_p": 0.9})
    p = store.create_profile(
        "m1", "derived", source_template="base", settings={"temperature": 0.2}
    )
    assert p.settings == {"temperature": 0.2, "top_p": 0.9}
    assert p.source_template == "base"


def test_an_unknown_source_template_is_an_error(store):
    with pytest.raises(ProfileError):
        store.create_profile("m1", "x", source_template="nope")


def test_update_merges_and_absent_fields_are_left_alone(store):
    """UpdateProfileRequest omits its nil fields, so a settings-only update
    must not clear the display name or reset exposure set elsewhere."""
    store.create_profile("m1", "fast", display_name="Fast",
                         settings={"temperature": 0.1}, expose_as_model=True)
    updated = store.update_profile("m1", "fast", settings={"temperature": 0.3})
    assert updated.display_name == "Fast"
    assert updated.expose_as_model is True
    assert updated.settings == {"temperature": 0.3}


def test_rename_moves_the_profile(store):
    store.create_profile("m1", "old", settings={"top_k": 5})
    store.update_profile("m1", "old", new_name="new")
    names = [p.name for p in store.list_profiles("m1")]
    assert names == ["new"]
    assert store.get_profile("m1", "new").settings == {"top_k": 5}


def test_delete_reports_whether_anything_went(store):
    store.create_profile("m1", "fast")
    assert store.delete_profile("m1", "fast") is True
    assert store.delete_profile("m1", "fast") is False


def test_profiles_survive_a_restart(store, tmp_path):
    store.create_profile("m1", "fast", settings={"temperature": 0.1})
    assert ProfileStore(tmp_path).get_profile("m1", "fast").settings == {
        "temperature": 0.1
    }


def test_a_corrupt_file_does_not_stop_the_server(tmp_path):
    path = tmp_path / ".bwr" / "profiles.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    s = ProfileStore(tmp_path)
    assert {t.name for t in s.list_templates()} >= {"precise"}
    assert s.list_profiles("m1") == []


# -- engine vs sampling ------------------------------------------------------


def test_engine_fields_are_flagged(store):
    sampling = store.create_profile("m1", "s", settings={"temperature": 0.1})
    engine = store.create_profile("m1", "e", settings={"n_ctx": 8192})
    assert sampling.has_engine_fields is False
    assert engine.has_engine_fields is True


def test_only_sampling_settings_reach_an_overlay(store):
    """An exposed profile that also sets n_ctx cannot apply it per request.
    Ignoring it beats failing a request over a field the caller cannot see."""
    store.create_profile(
        "m1", "mixed",
        settings={"temperature": 0.2, "n_ctx": 8192},
        expose_as_model=True,
    )
    base, overlay = store.resolve_overlay("m1:mixed")
    assert base == "m1"
    assert overlay == {"temperature": 0.2}


# -- exposure ----------------------------------------------------------------


def test_exposed_ids_are_listed(store):
    store.create_profile("m1", "fast", expose_as_model=True)
    store.create_profile("m1", "hidden", expose_as_model=False)
    assert [eid for eid, _b, _p in store.exposed()] == ["m1:fast"]


def test_an_unexposed_profile_does_not_resolve(store):
    store.create_profile("m1", "hidden", settings={"temperature": 0.2})
    # Unchanged id and empty overlay: the pool then rejects "m1:hidden" as
    # the unknown model it is, rather than silently serving m1.
    assert store.resolve_overlay("m1:hidden") == ("m1:hidden", {})


def test_a_plain_id_passes_through(store):
    assert store.resolve_overlay("m1") == ("m1", {})


def test_split_exposed_handles_ids_without_a_profile():
    assert split_exposed("plain") == ("plain", None)
    assert split_exposed("a:b") == ("a", "b")
    # Splits on the LAST separator, so a base id containing one survives.
    assert split_exposed("host:port:name") == ("host:port", "name")


def test_exposed_id_is_the_inverse_of_split():
    assert split_exposed(exposed_id("m1", "fast")) == ("m1", "fast")


# -- no model directory ------------------------------------------------------


def test_without_a_model_dir_the_store_works_in_memory_and_writes_nothing():
    """A single-model server has nowhere to anchor a file. Profiles still
    behave, they just do not persist -- better than refusing to construct."""
    s = ProfileStore(None)
    assert s.path is None
    s.create_profile("m1", "fast", settings={"top_p": 0.5})
    assert s.get_profile("m1", "fast").settings == {"top_p": 0.5}
