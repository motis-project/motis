import React from 'react';

import moment from 'moment';

import { Station, TripId } from '../Types/Connection';
import { classToId } from './ConnectionRender';
import { Trip } from '../Types/RailvizStationEvent';
import { Translations } from '../App/Localization';
import { SubOverlayEvent } from '../Types/EventHistory';


interface TripSearch {
    'translation': Translations,
    'trip': {first_station: Station, trip_info: Trip},
    'subOverlayContent': SubOverlayEvent[],
    'setTrainSelected': React.Dispatch<React.SetStateAction<TripId>>,
    'setSubOverlayContent': React.Dispatch<React.SetStateAction<SubOverlayEvent[]>>
}


export const TripSearch: React.FC<TripSearch> = (props) => {

    return (
        <div className='trip' key={`tripSearch ${props.trip.trip_info.id.time}`}>
            <div className='trip-train'>
                <span>
                    <div    className={`train-box train-class-${props.trip.trip_info.transport.clasz} with-tooltip`} 
                            data-tooltip={`${props.translation.connections.provider}: ${props.trip.trip_info.transport.provider}\n${props.translation.search.trainNr}: ${props.trip.trip_info.id.train_nr}`}
                            onClick={() => {
                                props.setTrainSelected(props.trip.trip_info.id);
                                props.setSubOverlayContent([...props.subOverlayContent, {id: 'tripView', train: props.trip.trip_info.id}]);
                            }}>
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
                <div    className='station' 
                        title={props.trip.first_station.name}
                        onClick={() => {
                            props.setSubOverlayContent([...props.subOverlayContent, {id: 'stationEvent', station: props.trip.first_station, stationTime: moment.unix(props.trip.trip_info.id.time)}]);
                        }}>
                    {props.trip.first_station.name}
                </div>
                <div className='direction' title={props.trip.trip_info.transport.direction}>
                    <i className='icon'>arrow_forward</i>
                    {props.trip.trip_info.transport.direction}
                </div>
            </div>
        </div>
    )
}