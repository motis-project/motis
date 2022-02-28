import moment from 'moment';
import React, { useEffect, useState } from 'react';
import { JourneyRender } from './ConnectionRender';
import { Connection, Station, TripId, TripViewConnection } from './ConnectionTypes';

const displayTime = (posixTime) => {
    let today = new Date(posixTime * 1000);
    let h = String(today.getHours());
    let m = String(today.getMinutes()).padStart(2, '0');
    return h + ':' + m;
}

const displayDuration = (posixTime) => {
    let today = new Date(posixTime * 1000);
    let h = String(today.getUTCHours());
    let m = String(today.getUTCMinutes()).padStart(2, '0');
    if (h === '0') {
        return m + 'min';
    } else {
        return h + 'h ' + m + 'min';
    }
}

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

export const FetchTrainData: React.FC<{ 'subOverlayHidden': Boolean, 'setSubOverlayHidden': React.Dispatch<React.SetStateAction<Boolean>>, 'trainSelected': TripId, 'setTrainSelected': React.Dispatch<React.SetStateAction<TripId>> }> = (props) => {

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
                });
        }
    }, [props.subOverlayHidden]);

    return (
        <div className='connection-details trip-view'>
            <div className='connection-info'>
                <div className='header'>
                    <div className='back'><i className='icon' onClick={() => props.setSubOverlayHidden(true)}>arrow_back</i></div>
                    <div className='details'>
                        <div className='date'>22.1.2022</div>
                        <div className='connection-times'>
                            <div className='times'>
                                <div className='connection-departure'>{displayTime(time)}</div>
                                <div className='connection-arrival'>{displayTime(targetTime)}</div>
                            </div>
                            <div className='locations'>
                                <div>{trainConnection.stops[0].station.name}</div>
                                <div>{trainConnection.stops[trainConnection.stops.length - 1].station.name}</div>
                            </div>
                        </div>
                        <div className='summary'><span className='duration'><i className='icon'>schedule</i>{displayDuration(targetTime - time)}</span><span
                            className='interchanges'><i className='icon'>transfer_within_a_station</i>Keine Umstiege</span></div>
                    </div>
                    <div className='actions'></div>
                </div>
            </div>
            <div className='connection-journey' id='sub-connection-journey'>
                <JourneyRender connection={trainConnection} setSubOverlayHidden={props.setSubOverlayHidden} setTrainSelected={props.setTrainSelected} />
            </div>
        </div>
    )

}