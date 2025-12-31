from datetime import datetime

from speed_test import speed_test


def main():
    # MATLAB: time_today = datestr(now, 'mmddyyyy');
    time_today = datetime.now().strftime("%m%d%Y")

    # MATLAB:
    # traj_type = 'circle';
    # speed_test
    traj_type = "circle"
    speed_test(traj_type=traj_type, time_today=time_today)

    # MATLAB:
    # traj_type = 'infty';
    # speed_test
    traj_type = "infty"
    speed_test(traj_type=traj_type, time_today=time_today)


if __name__ == "__main__":
    main()
