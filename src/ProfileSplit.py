import pandas as pd

# for spliting original dataset, group by profile
# output .csv file in folder "profile_data"
def main(path):
    orig = pd.read_csv(path)
    gps = orig.groupby("profile_id")
    for k, gp in gps:
        gp.to_csv('profile_data/Profile_' + str(k) + '.csv')

if __name__ == '__main__':
    main("pmsm_temperature_data.csv")