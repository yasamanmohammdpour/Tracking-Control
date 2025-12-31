from datetime import datetime
from uncertain_test_saferegion import uncertain_test_saferegion


def main():
    # MATLAB: time_today = datestr(now, 'mmddyyyy');
    time_today = datetime.now().strftime("%m%d%Y")

    # MATLAB:
    # traj_type = 'infty';
    # uncertain_test_saferegion
    traj_type = "infty"

    uncertain_test_saferegion(traj_type=traj_type, time_today=time_today)


if __name__ == "__main__":
    main()
