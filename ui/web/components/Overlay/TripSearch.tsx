import React from 'react';

import moment from 'moment';

import { Station } from '../Types/Connection';
import { classToId } from './ConnectionRender';
import { Trip } from '../Types/RailvizStationEvent';
import { Translations } from '../App/Localization';

export const TripSearch: React.FC<{'translation': Translations, 'trip': {first_station: Station, trip_info: Trip}}> = (props) => {
    
    console.log(props.trip.trip_info.id.time)
    console.log(moment(props.trip.trip_info.id.time))

    return (
        <div className='trip'>
            <div className='trip-train'>
                <span>
                    <div className={`train-box train-class-${props.trip.trip_info.transport.clasz} with-tooltip`} data-tooltip={`${props.translation.connections.provider}: ${props.trip.trip_info.transport.provider} ${props.translation.search.trainNr}: ${props.trip.trip_info.id.train_nr}`}>
                        <svg className='train-icon'>
                            <use xlinkHref={classToId({move: props.trip.trip_info.transport, move_type: 'Transport'})}></use>
                        </svg>
                        <span className='train-name'>{props.trip.trip_info.transport.name}</span>
                    </div>
                </span>
            </div>
            <div className='trip-time'>
                <div className='time'>{moment.unix(props.trip.trip_info.id.time).format('HH:mm')}</div>
                <div className='date'>{moment.unix(props.trip.trip_info.id.time).format('D.M.')}</div>
            </div>
            <div className='trip-first-station'>
                <div className='station' title={props.trip.first_station.name}>{props.trip.first_station.name}</div>
                <div className='direction' title={props.trip.trip_info.transport.direction}>
                    <i className='icon'>arrow_forward</i>
                    {props.trip.trip_info.transport.direction}
                </div>
            </div>
        </div>
    )
}