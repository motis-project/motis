import React from 'react';

import moment from 'moment';

import { Station, TripId } from '../Types/Connection';
import { classToId } from './ConnectionRender';
import { Trip } from '../Types/RailvizStationEvent';
import { Translations } from '../App/Localization';
import { Address } from '../Types/SuggestionTypes';

export const TripSearch: React.FC<{'translation': Translations, 'trip': {first_station: Station, trip_info: Trip}, 'setTrainSelected': React.Dispatch<React.SetStateAction<TripId>>, 'setStationSearch': React.Dispatch<React.SetStateAction<Station | Address>>, 'setSubOverlayDate': React.Dispatch<React.SetStateAction<moment.Moment>>}> = (props) => {

    return (
        <div className='trip'>
            <div className='trip-train'>
                <span>
                    <div    className={`train-box train-class-${props.trip.trip_info.transport.clasz} with-tooltip`} 
                            data-tooltip={`${props.translation.connections.provider}: ${props.trip.trip_info.transport.provider} ${props.translation.search.trainNr}: ${props.trip.trip_info.id.train_nr}`}
                            onClick={() => {
                                props.setTrainSelected(props.trip.trip_info.id);
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
                            props.setStationSearch(null);
                            props.setStationSearch(props.trip.first_station);
                            props.setSubOverlayDate(moment.unix(props.trip.trip_info.id.time));
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