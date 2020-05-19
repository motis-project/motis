#!/bin/env python3

import argparse
import json
import requests
import time


def make_query(dep, arr, t = int(time.time())):
  return {
    'destination': {
      'type': 'Module',
      'target': '/rt/single',
    }, 
    'content_type': 'Message',
    'content': {
      'earliest': 0,
      'latest': 0,
      'timestamp': 0,
      'content_type': 'AdditionMessage',
      'content': {
          'trip_id': {
            'station_id': dep,
            'service_num': 77777,
            'schedule_time': t - 2 * 60,
            'trip_type': 'Additional'
          },
          'events': [{
            'base': {
              'station_id': dep,
              'service_num': 77777,
              'line_id': '',
              'type': 'DEP',
              'schedule_time': t - 2 * 60
            },
            'category': 'TGV',
            'track': '9'
          }, {
            'base': {
              'station_id': arr,
              'service_num': 77777,
              'line_id': '',
              'type': 'ARR',
              'schedule_time': t + 20 * 60
            },
            'category': 'TGV',
            'track': '9'
          }]
      }
    }
  }


if __name__ == '__main__':
  p = argparse.ArgumentParser(description='adds a single additional train', 
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument('--url', '-u', nargs='?', default='http://localhost:8080', help='MOTIS url')
  p.add_argument('--dep', '-d', nargs='?', default='8000105', help='dep station id')
  p.add_argument('--arr', '-a', nargs='?', default='8000068', help='arr station id')

  args = p.parse_args()

  r = requests.post(args.url, json = make_query(args.dep, args.arr))
  print(r.status_code)
  print(r.text)
