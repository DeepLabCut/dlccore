#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/dlccore
#
# Contributors:
# https://github.com/DeepLabCut/dlccore/graphs/contributors
#
# Licensed under GNU Lesser General Public License v3.0
#


from dlccore.project.validation.baseline import (
    iter_baseline_check_registrations,
)


def test_baseline_check_registrations_are_well_formed() -> None:
    registrations = iter_baseline_check_registrations()
    codes = tuple(registration.code for registration in registrations)

    assert registrations
    assert len(codes) == len(set(codes))

    for registration in registrations:
        assert registration.code
        assert registration.description
        assert callable(registration.validator)
