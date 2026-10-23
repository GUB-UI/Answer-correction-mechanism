from pathlib import Path

from packaging.requirements import Requirement


def test_root_requirements_cover_harness_dependencies():
    root = Path(__file__).resolve().parents[2]
    def requirements(path):
        return {req.name.lower().replace('_', '-'): req for line in path.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith('#') for req in [Requirement(line)]}
    declared = requirements(root / 'requirements.txt')
    for name, dependency in requirements(root / 'harness_v2/requirements.txt').items():
        assert name in declared, name
        # Harness currently uses lower bounds only; root must preserve each bound.
        for bound in dependency.specifier:
            assert str(bound) in {str(item) for item in declared[name].specifier}, (name, bound)
