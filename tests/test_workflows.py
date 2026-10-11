"""Recording and finding workflows — the "record once" half."""

import yaml

from src.plan import Plan, Step
from src import workflows


class TestSaveAndLoad:
    def test_round_trips(self, simple_plan, wf_dir):
        path = workflows.save(simple_plan, "say hello", wf_dir)
        loaded = workflows.load(path)
        assert loaded.goal == "say hello"
        assert loaded.plan == simple_plan

    def test_filename_comes_from_the_goal(self, simple_plan, wf_dir):
        path = workflows.save(simple_plan, "Say Hello, World!", wf_dir)
        assert path.name == "say-hello-world.yaml"

    def test_long_goals_are_truncated(self, simple_plan, wf_dir):
        path = workflows.save(simple_plan, "word " * 50, wf_dir)
        assert len(path.stem) <= 60

    def test_yaml_is_human_editable(self, simple_plan, wf_dir):
        # These files are meant to be opened and tweaked by hand.
        raw = yaml.safe_load(workflows.save(simple_plan, "say hello", wf_dir).read_text())
        assert raw["steps"][0]["args"]["command"] == "echo hello"
        assert raw["controller"] == "shell"

    def test_operator_goal_survives_the_plans_own_goal(self, wf_dir):
        # Regression: plan.to_dict() also carries a "goal" (the model's
        # rephrasing) and used to overwrite the operator's wording on save,
        # which broke find() — the file was named one thing and matched another.
        plan = Plan(controller="shell", goal="MODEL REPHRASING", steps=[Step("run", {})])
        path = workflows.save(plan, "what the operator asked", wf_dir)
        assert yaml.safe_load(path.read_text())["goal"] == "what the operator asked"
        assert workflows.find("what the operator asked", wf_dir) is not None


class TestFind:
    def test_exact_goal_matches(self, simple_plan, wf_dir):
        workflows.save(simple_plan, "count the files", wf_dir)
        assert workflows.find("count the files", wf_dir) is not None

    def test_matching_ignores_case_and_punctuation(self, simple_plan, wf_dir):
        workflows.save(simple_plan, "Count the files!", wf_dir)
        assert workflows.find("count the files", wf_dir) is not None

    def test_close_wording_still_matches(self, simple_plan, wf_dir):
        workflows.save(simple_plan, "count the python files", wf_dir)
        assert workflows.find("count the python files now", wf_dir) is not None

    def test_unrelated_goal_does_not_match(self, simple_plan, wf_dir):
        # A wrong match replays the wrong actions, so finding nothing is the
        # safer failure: it just costs one planning call.
        workflows.save(simple_plan, "count the python files", wf_dir)
        assert workflows.find("send an email to my boss", wf_dir) is None

    def test_partial_overlap_below_threshold_does_not_match(self, simple_plan, wf_dir):
        workflows.save(simple_plan, "delete every temporary file in the cache", wf_dir)
        assert workflows.find("delete the database", wf_dir) is None

    def test_no_workflows_means_no_match(self, wf_dir):
        assert workflows.find("anything", wf_dir) is None


class TestRobustness:
    def test_empty_file_is_skipped_not_fatal(self, simple_plan, wf_dir):
        # src/workflows/test.yaml is empty in the repo; it must not break startup.
        (wf_dir / "empty.yaml").write_text("")
        workflows.save(simple_plan, "real one", wf_dir)
        assert len(workflows.all_workflows(wf_dir)) == 1

    def test_malformed_yaml_is_skipped(self, simple_plan, wf_dir):
        (wf_dir / "broken.yaml").write_text("{{{ not yaml")
        workflows.save(simple_plan, "real one", wf_dir)
        assert len(workflows.all_workflows(wf_dir)) == 1

    def test_yaml_missing_steps_is_skipped(self, wf_dir):
        (wf_dir / "partial.yaml").write_text("goal: no steps here\n")
        assert workflows.all_workflows(wf_dir) == []

    def test_non_yaml_files_are_ignored(self, simple_plan, wf_dir):
        (wf_dir / "notes.txt").write_text("hello")
        workflows.save(simple_plan, "real one", wf_dir)
        assert len(workflows.all_workflows(wf_dir)) == 1


class TestRunCounter:
    def test_counts_replays(self, simple_plan, wf_dir):
        workflows.save(simple_plan, "say hello", wf_dir)
        found = workflows.find("say hello", wf_dir)
        workflows.record_run(found)
        assert workflows.find("say hello", wf_dir).runs == 1

    def test_counter_persists_across_loads(self, simple_plan, wf_dir):
        workflows.save(simple_plan, "say hello", wf_dir)
        for _ in range(3):
            workflows.record_run(workflows.find("say hello", wf_dir))
        assert workflows.find("say hello", wf_dir).runs == 3

    def test_recording_a_run_keeps_the_goal_and_steps(self, simple_plan, wf_dir):
        workflows.save(simple_plan, "say hello", wf_dir)
        workflows.record_run(workflows.find("say hello", wf_dir))
        again = workflows.find("say hello", wf_dir)
        assert again.goal == "say hello"
        assert again.plan.steps == simple_plan.steps
