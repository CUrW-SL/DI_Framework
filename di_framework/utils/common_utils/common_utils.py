import datetime as dt


def datetime_lk_to_utc(timestamp_lk,  shift_mins=0):
    return timestamp_lk - dt.timedelta(hours=5, minutes=30 + shift_mins)


def datetime_utc_to_lk(timestamp_utc, shift_mins=0):
    return timestamp_utc + dt.timedelta(hours=5, minutes=30 + shift_mins)