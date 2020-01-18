#! /usr/bin/env python3

'''
mamikhai@cisco.com
20200111
this module functions imported into monitor.py
read and construct DataFrames for input and validation.
'''

from datetime import datetime
import pandas as pd
from influxdb import DataFrameClient
from constants import *

pd.options.display.max_rows = None
pd.options.display.max_columns = None

client = DataFrameClient(host='localhost', port=8086, database=target_db)

def read_data(field_key, measurement_name, condition1, condition2, condition3, limit, label):
  if is_demo:
    shift = datetime.now() - datetime(2020, 1, 9)
    time_shift = str(' - ' + str(shift.days) + 'd ')
    condition2 = condition2 + time_shift
    # print(condition2)
    # print('time_shift = ', time_shift)
  query_db = str('SELECT "%s" FROM "%s" WHERE %s AND %s AND %s LIMIT %d ' % (field_key, measurement_name, condition1, condition2, condition3, limit+1))
  data_db = client.query(query_db)
  data_df = pd.DataFrame(data_db[str(measurement_name)])
  data_df.columns = [label]
  data_df.reset_index(drop=True, inplace=True)
  data_df.fillna(method='ffill', inplace=True)
  data_df.fillna(method='bfill', inplace=True)
  data_df -= data_df.min()
  data_df.drop(data_df.index[0], inplace=True)
  # data_df = data_df.sub(data_df.shift(fill_value=0))
  # print('\n', query_db, '\n', data_df.describe())
  return data_df

def read_last_target(record_count, label_prefix, previous, verbose=True):
    # read last 1h of tunnel interfaces egress counts
    for interface in tunnel_ifs:
        label = str(label_prefix + interface[-7:])
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1h - 1m', 'time <= now()', record_count, label) 
        if interface == tunnel_ifs[0]:
            validate_target = read_if
        else:
            validate_target = pd.concat([validate_target, read_if], axis=1, sort=False)
    validate_target.fillna(method='ffill', inplace=True)
    '''
    # read last 1h of physical core interfaces egress and ingress counts
    for interface in physical_ifs:
        label = str(label_prefix + 's' + interface[-6:])
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1h - 1m', 'time <= now()', record_count, label)
        validate_target = pd.concat([validate_target, read_if], axis=1, sort=False)
    
        label = str(label_prefix + 'r' + interface[-6:])
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-received', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1h - 1m', 'time <= now()', record_count, label)
        validate_target = pd.concat([validate_target, read_if], axis=1, sort=False)
    '''
    if verbose:
        print('\nvalidation target')
        print(validate_target.describe())
    return validate_target

def read_validate(record_count, label_prefix, previous, verbose=True):
    # read previous 1h, same hour previous day, same hour a week ago, of tunnel interfaces egress counts
    for interface in tunnel_ifs:
        query_if = str('("interface-name" = \'%s\')' % (interface))
        label = str(label_prefix + interface[-7:] + "_previous")
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 2h - 1m', 'time <= now()', record_count, label)    # - 90m for 30 minute overlap
        if interface == tunnel_ifs[0]:
            validate = read_if
        else:
            validate = pd.concat([validate, read_if], axis=1, sort=False)
        label = str(label_prefix + interface[-7:] + "_1d")
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - {} - 1h - 1m'.format(previous), 'time <= now()', record_count, label)
        validate = pd.concat([validate, read_if], axis=1, sort=False)
        label = str(label_prefix + interface[-7:] + "_1w")
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 2w - 1h - 1m', 'time <= now()', record_count, label)
        validate = pd.concat([validate, read_if], axis=1, sort=False)
        label = str(label_prefix + interface[-7:] + "_2w")
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 3w - 1h - 1m', 'time <= now()', record_count, label)
        validate = pd.concat([validate, read_if], axis=1, sort=False)
    validate.fillna(method='ffill', inplace=True)
    
    '''
    # read previous 1h, same hour previous day, same hour a week ago, of physical core interfaces egress and ingress counts
    for interface in physical_ifs:
        label = str(label_prefix + 's' + interface[-6:] + "_previous")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 2h - 1m', 'time <= now()', record_count, label)
        validate = pd.concat([validate, read_if], axis=1, sort=False)
        label = str(label_prefix + 'r' + interface[-6:] + "_previous")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-received', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 2h - 1m', 'time <= now()', record_count, label)
        validate = pd.concat([validate, read_if], axis=1, sort=False)
        label = str(label_prefix + 's' + interface[-6:] + "_1d")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1d - 1h - 1m', 'time <= now()', record_count, label)
        validate = pd.concat([validate, read_if], axis=1, sort=False)
        label = str(label_prefix + 'r' + interface[-6:] + "_1d")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-received', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1d - 1h - 1m', 'time <= now()', record_count, label)
        validate = pd.concat([validate, read_if], axis=1, sort=False)
        label = str(label_prefix + 's' + interface[-6:] + "_1w")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1w - 1h - 1m', 'time <= now()', record_count, label)
        validate = pd.concat([validate, read_if], axis=1, sort=False)
        label = str(label_prefix + 'r' + interface[-6:] + "_1w")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-received', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1w - 1h - 1m', 'time <= now()', record_count, label)
        validate = pd.concat([validate, read_if], axis=1, sort=False)
    '''
    if verbose:
        print('\nvalidation data')
        print(validate.describe())
    return validate

def read_train_target(record_count, label_prefix, previous, verbose=True):
    # read previous day's same 1h of tunnel interfaces egress counts
    for interface in tunnel_ifs:
        label = str(label_prefix + interface[-7:])
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - {} - 1h - 1m'.format(previous), 'time <= now()', record_count, label)    # - 90m for 30 minute overlap
        if interface == tunnel_ifs[0]:
            train_target = read_if
        else:
            train_target = pd.concat([train_target, read_if], axis=1, sort=False)
    train_target.fillna(method='ffill', inplace=True)

    '''
    # read previous day's same 1h of physical core interfaces egress and ingress counts
    for interface in physical_ifs:
        label = str(label_prefix + 's' + interface[-6:])
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1d - 1h - 1m', 'time <= now()', record_count, label)
        train_target = pd.concat([train_target, read_if], axis=1, sort=False)
        label = str(label_prefix + 'r' + interface[-6:])
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-received', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1d - 1h - 1m', 'time <= now()', record_count, label)
        train_target = pd.concat([train_target, read_if], axis=1, sort=False)
    '''
    if verbose:
        print('\ntraining target')
        print(train_target.describe())
    return train_target

def read_train_target_long(record_count, label_prefix, previous, verbose=True):
    # read previous day's 24h of tunnel interfaces egress counts
    for interface in tunnel_ifs:
        label = str(label_prefix + interface[-7:])
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - {} - 12h - 30m - 1m'.format(previous), 'time <= now()', record_count, label)
        if interface == tunnel_ifs[0]:
            train_target = read_if
        else:
            train_target = pd.concat([train_target, read_if], axis=1, sort=False)
    
    train_target.fillna(method='ffill', inplace=True)
    '''
    # read previous day's 24h of physical core interfaces egress and ingress counts
    for interface in physical_ifs:
        label = str(label_prefix + 's' + interface[-6:])
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1d - 12h - 30m - 1m', 'time <= now()', record_count, label)
        train_target = pd.concat([train_target, read_if], axis=1, sort=False)
        label = str(label_prefix + 'r' + interface[-6:])
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-received', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1d - 12h - 30m - 1m', 'time <= now()', record_count, label)
        train_target = pd.concat([train_target, read_if], axis=1, sort=False)
    '''
    if verbose:
        print('\ntraining target')
        print(train_target.describe())
    return train_target

def read_train(record_count, label_prefix, previous, verbose=True):
    # read previous 1h to target (last day's same 1h), day earlier, a week earlier, of tunnel interfaces egress counts
    for interface in tunnel_ifs:
        query_if = str('("interface-name" = \'%s\')' % (interface))
        label = str(label_prefix + interface[-7:] + "_previous")
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - {} - 2h - 1m'.format(previous), 'time <= now()', record_count, label)
        if interface == tunnel_ifs[0]:
            train = read_if
        else:
            train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + interface[-7:] + "_1d")
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - {} - 1d - 1h - 1m'.format(previous), 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + interface[-7:] + "_1w")
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1w - {} - 1h - 1m'.format(previous), 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + interface[-7:] + "_2w")
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 2w - {} - 1h - 1m'.format(previous), 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
    
    train.fillna(method='ffill', inplace=True)
    '''
    # read previous 1h to target (last day's same 1h), day earlier, a week earlier, of physical core interfaces egress and ingress counts
    for interface in physical_ifs:
        label = str(label_prefix + 's' + interface[-6:] + "_previous")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1d - 2h - 1m', 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + 'r' + interface[-6:] + "_previous")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-received', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1d - 2h - 1m', 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + 's' + interface[-6:] + "_1d")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 2d - 1h - 1m', 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + 'r' + interface[-6:] + "_1d")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-received', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 2d - 1h - 1m', 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + 's' + interface[-6:] + "_1w")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1w - 1d - 1h - 1m', 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + 'r' + interface[-6:] + "_1w")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-received', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1w - 1d - 1h - 1m', 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
    '''
    return train

def read_train_long(record_count, label_prefix, previous, verbose=True):
    '''
    global feature_mean
    global feature_std
    global feature_max
    '''
    # read 24h shifted 1h to target,  previous 24h to target, same day a week earlier, of tunnel interfaces egress counts
    for interface in tunnel_ifs:
        query_if = str('("interface-name" = \'%s\')' % (interface))
        label = str(label_prefix + interface[-7:] + "_previous")
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - {} - 13h - 30m - 1m'.format(previous), 'time <= now()', record_count, label)   # was - 20h - 30m
        if interface == tunnel_ifs[0]:
            train = read_if
        else:
            train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + interface[-7:] + "_1d")
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - {} - 1d - 12h - 30m - 1m'.format(previous), 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + interface[-7:] + "_1w")
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1w - {} - 12h - 30m - 1m'.format(previous), 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + interface[-7:] + "_2w")
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 2w - {} - 12h - 30m - 1m'.format(previous), 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
    train.fillna(method='ffill', inplace=True)

    '''
    # read 24h shifted 1h to target, previous 24h to target, same day a week earlier, of physical core interfaces egress and ingress counts
    for interface in physical_ifs:
        label = str(label_prefix + 's' + interface[-6:] + "_previous")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1d - 13h - 30m - 1m', 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + 'r' + interface[-6:] + "_previous")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-received', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1d - 13h - 30m - 1m', 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + 's' + interface[-6:] + "_1d")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 2d - 12h - 30m - 1m', 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + 'r' + interface[-6:] + "_1d")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-received', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 2d - 12h - 30m - 1m', 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + 's' + interface[-6:] + "_1w")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-sent', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1w - 1d - 12h - 30m - 1m', 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
        label = str(label_prefix + 'r' + interface[-6:] + "_1w")
        query_if = str('("interface-name" = \'%s\')' % (interface))
        read_if = read_data('bytes-received', 'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/generic-counters', \
                query_if, 'time >= now() - 1w - 1d - 12h - 30m - 1m', 'time <= now()', record_count, label)
        train = pd.concat([train, read_if], axis=1, sort=False)
    if feature_mean == 0:
        feature_mean = train.mean().mean()
        print('feature mean: ', feature_mean)
    if feature_std == 0:
        feature_std = train.std().mean()
        print('feature std: ', feature_std)
    if feature_max == 0:
        feature_max = train.max().mean() / 24 # The mean max per 1 hour
        print('feature max: ', feature_max)
    '''
    if verbose:
        print('\ntraining long data')
        print(train.describe())
    return train

