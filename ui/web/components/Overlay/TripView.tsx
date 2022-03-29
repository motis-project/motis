import React, { useEffect, useState } from 'react';

import moment from 'moment';

import { JourneyRender, duration} from './Journey';
import { Connection, TripId, TripViewConnection } from '../Types/Connection';
import { Translations } from '../App/Localization';
import { getMapFilter } from './Overlay';


const getTrainConnection = (lineId: string, stationId: string, targetStationId: string, targetTime: number, time: number, trainNr: number) => {
    return {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            destination: { type: "Module", target: "/trip_to_connection" },
            content_type: 'TripId',
            content: { line_id: lineId, station_id: stationId, target_station_id: targetStationId, target_time: targetTime, time: time, train_nr: trainNr }
        })
    };
};


export const TripView: React.FC<{ 'subOverlayHidden': Boolean, 'setSubOverlayHidden': React.Dispatch<React.SetStateAction<Boolean>>, 'trainSelected': TripId, 'setTrainSelected': React.Dispatch<React.SetStateAction<TripId>>, 'detailViewHidden': Boolean, 'translation': Translations, 'displayDate': moment.Moment, 'mapFilter': any}> = (props) => {

    const [lineId, setLineId] = useState<string>(props.trainSelected.line_id);

    const [stationId, setStationId] = useState<string>(props.trainSelected.station_id);

    const [targetStationId, setTargetStationId] = useState<string>(props.trainSelected.target_station_id);

    const [targetTime, setTargetTime] = useState<number>(props.trainSelected.target_time);

    const [time, setTime] = useState<number>(props.trainSelected.time);

    const [train, setTrain] = useState<TripId>(props.trainSelected);

    const [trainConnection, setTrainConnection] = useState<Connection>();

    useEffect(() => {
        if (props.trainSelected !== undefined) {
            let requestURL = 'https://europe.motis-project.de/?elm=tripRequest';
            fetch(requestURL, getTrainConnection(lineId, stationId, targetStationId, targetTime, time, train.train_nr))
                .then(res => res.json())
                .then((res: TripViewConnection) => {
                    console.log('Trip Request successful');
                    console.log(res);
                    setTrainConnection(res.content);
                    window.portEvents.pub('mapSetDetailFilter', getMapFilter(res.content));
                });
        }
    }, [props.subOverlayHidden]);

    return (
        (trainConnection === undefined) ?
            <></> 
            :
            <div className='connection-details trip-view'>
                <div className='connection-info'>
                    <div className='header'>
                        <div className='back' onClick={() => {props.setTrainSelected(undefined); window.portEvents.pub('mapSetDetailFilter', props.mapFilter)}}><i className='icon'>arrow_back</i></div>
                        <div className='details'>
                            <div className='date'>{moment.unix(props.displayDate.unix()).format(props.translation.dateFormat)}</div>
                            <div className='connection-times'>
                                <div className='times'>
                                    <div className='connection-departure'>{moment.unix(time).format('HH:mm')}</div>
                                    <div className='connection-arrival'>{moment.unix(targetTime).format('HH:mm')}</div>
                                </div>
                                <div className='locations'>
                                    <div>{trainConnection.stops[0].station.name}</div>
                                    <div>{trainConnection.stops[trainConnection.stops.length - 1].station.name}</div>
                                </div>
                            </div>
                            <div className='summary'><span className='duration'><i className='icon'>schedule</i>{duration(time, targetTime)}</span>
                            <span className='interchanges'><i className='icon'>transfer_within_a_station</i>{props.translation.connections.interchanges(0)}</span></div>
                        </div>
                        <div className='actions'></div>
                    </div>
                </div>
                <div className='connection-journey' id='sub-connection-journey'>
                    <JourneyRender connection={trainConnection} setSubOverlayHidden={props.setSubOverlayHidden} setTrainSelected={props.setTrainSelected} detailViewHidden={props.detailViewHidden} translation={props.translation} comingFromTripView={true}/>
                </div>
            </div>
    )
}