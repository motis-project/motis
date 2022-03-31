import React, { useEffect, useState } from 'react';

import moment from 'moment';

import { JourneyRender, duration} from './Journey';
import { Connection, Station, Transport, TripId, TripViewConnection } from '../Types/Connection';
import { Translations } from '../App/Localization';
import { getMapFilter } from './Overlay';
import { Address } from '../Types/SuggestionTypes';
import { SubOverlayEvent } from '../Types/EventHistory';
import { getFromLocalStorage } from '../App/LocalStorage';


interface TripView {
    'trainSelected': TripId | Connection,
    'overlayTripView': Connection,
    'translation': Translations,
    'mapFilter': any
    'setTrainSelected': React.Dispatch<React.SetStateAction<TripId>>,
    'setTripViewHidden': React.Dispatch<React.SetStateAction<Boolean>>,
    'subOverlayContent': SubOverlayEvent[], 
    'setSubOverlayContent': React.Dispatch<React.SetStateAction<SubOverlayEvent[]>>,
}

// helperfunction to get number of Interchanges in this trip
const getTransportCountString = (transports: Transport[], translation: Translations) => {
    let count = 0;
    for (let index = 0; index < transports.length; index++) {
        if (transports[index].move_type === 'Transport' && index > 0) {
            count++
        }
    }
    return translation.connections.interchanges(count);
}


// Helperfunction to differentiate objects that can be either a TripId or a Connection
const isTripId = (t: TripId | Connection): t is TripId => {
    return (t as TripId).line_id !== undefined
}


const getTrainConnection = (lineId: string, stationId: string, targetStationId: string, targetTime: number, time: number, trainNr: number) => {
    return {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            destination: { type: 'Module', target: '/trip_to_connection' },
            content_type: 'TripId',
            content: { line_id: lineId, station_id: stationId, target_station_id: targetStationId, target_time: targetTime, time: time, train_nr: trainNr }
        })
    };
};

// helperfunction to get the coords for each stop of a connection
export const getStationCoords = (connection: Connection) => {
    let coords = [];
    for(let i = 0; i < connection.stops.length; i++){
        let pos = [];
        pos.push(connection.stops[i].station.pos.lat);
        pos.push(connection.stops[i].station.pos.lng);
        coords.push(pos);
    }
    return {mapId: 'map', coords};
};

export const TripView: React.FC<TripView> = (props) => {

    const [trainConnection, setTrainConnection] = useState<Connection>(isTripId(props.trainSelected) ? undefined : props.trainSelected);

    const time = isTripId(props.trainSelected) ? props.trainSelected.time : props.trainSelected.stops[0].departure.time;

    const targetTime = isTripId(props.trainSelected) ? props.trainSelected.target_time : props.trainSelected.stops.at(-1).arrival.time;

    const [start, setStart] = useState<Station | Address>(getFromLocalStorage('motis.routing.from_location'));

    const [destination, setDestination] = useState<Station | Address>(getFromLocalStorage('motis.routing.to_location'));

    // everytime a new train is selected via trainbox, a tripRequest fetches the data needed to be displayed
    useEffect(() => {
        if (props.trainSelected && isTripId(props.trainSelected)) {
            let requestURL = 'https://europe.motis-project.de/?elm=tripRequest';
            fetch(requestURL, getTrainConnection(props.trainSelected.line_id, props.trainSelected.station_id, props.trainSelected.target_station_id, props.trainSelected.target_time, props.trainSelected.time, props.trainSelected.train_nr))
                .then(res => res.json())
                .then((res: TripViewConnection) => {
                    setTrainConnection(res.content);
                    window.portEvents.pub('mapSetDetailFilter', getMapFilter(res.content));
                    window.portEvents.pub('mapFitBounds', getStationCoords(res.content));
                });
        }
    }, [props.trainSelected]);

    return (
        (trainConnection === undefined) ?
            <></> 
            :
            <div className={`connection-details ${isTripId(props.trainSelected) ? 'trip-view' : ''}`}>
                <div className='connection-info'>
                    <div className='header'>
                        <div className='back' onClick={() => {
                                                if (isTripId(props.trainSelected)) {
                                                    let tmp = [...props.subOverlayContent];
                                                    tmp.pop();
                                                    props.setSubOverlayContent(tmp);
                                                    window.portEvents.pub('mapSetDetailFilter', props.mapFilter);
                                                    window.portEvents.pub('mapFitBounds', getStationCoords(props.overlayTripView));
                                                } else {
                                                    props.setTripViewHidden(true);
                                                }
                                                }}>
                            <i className='icon'>arrow_back</i>
                        </div>
                        <div className='details'>
                            <div className='date'>{moment.unix(time).format(props.translation.dateFormat)}</div>
                            <div className='connection-times'>
                                <div className='times'>
                                    <div className='connection-departure'>{moment.unix(time).format('HH:mm')}</div>
                                    <div className='connection-arrival'>{moment.unix(targetTime).format('HH:mm')}</div>
                                </div>
                                <div className='locations'>
                                    <div>{(trainConnection.stops[0].station.name === 'START') ? (start as Station).name : trainConnection.stops[0].station.name}</div>
                                    <div>{(trainConnection.stops[trainConnection.stops.length - 1].station.name === 'END') ? (destination as Station).name : trainConnection.stops[trainConnection.stops.length - 1].station.name}</div>
                                </div>
                            </div>
                            <div className='summary'>
                                <span className='duration'>
                                    <i className='icon'>schedule</i>
                                    {duration(time, targetTime)}
                                </span>
                                <span className='interchanges'>
                                    <i className='icon'>transfer_within_a_station</i>
                                    {getTransportCountString(trainConnection.transports, props.translation)}
                                </span>
                            </div>
                        </div>
                        {isTripId(props.trainSelected) ? 
                            <div className='actions' />
                            :
                            <div className='actions'>
                                <i className='icon'>save</i>
                                <i className='icon'>share</i>
                            </div>
                        }  
                    </div>
                </div>
                <div className='connection-journey' id={`${ isTripId(props.trainSelected) ? 'sub-' : ''}connection-journey`}>
                    <JourneyRender connection={trainConnection} setTrainSelected={props.setTrainSelected} translation={props.translation} subOverlayContent={props.subOverlayContent} setSubOverlayContent={props.setSubOverlayContent}/>
                </div>
            </div>
    )
}