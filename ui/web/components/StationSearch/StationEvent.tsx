import React, { useEffect, useState } from 'react';

import moment from 'moment';
import { RailVizStationRequest, StationEvents } from '../Types/RailvizStationEvent';
import { Station } from '../Types/Connection';
import { Address } from '../Types/SuggestionTypes';
import { Translations } from '../App/Localization';
import { classToId } from '../Overlay/ConnectionRender';

const getStationEvent = (byScheduleTime: boolean, direction: string, eventCount: number, stationID: string, time: number) => {
    return {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            destination: { type: 'Module', target: '/railviz/get_station' },
            content_type: 'RailVizStationRequest',
            content: { by_schedule_time: byScheduleTime, direction: direction, event_count: eventCount, station_id: stationID, time: time }
        })
    };
};


const stationEventDivGenerator = (eventStations: StationEvents) => {

    let events = eventStations.events;
    let divs = []

    for (let index = 0; index < events.length; index++) {
        divs.push(
            <div className='station-event'>
                <div className='event-time'>{events[index].event.time}</div>
                <div className='event-train'><span>
                    <div className={'train-box train-class-' + events[index].trips[0].transport.clasz + ' with-tooltip'} data-tooltip='Betreiber: DB Regio AG S-Bahn Rhein-Main Zugnummer: 35324'><svg className='train-icon'>
                        <use xlinkHref={classToId(events[index].trips[0].transport.clasz)}></use>
                    </svg><span className='train-name'>{events[index].trips[0].transport.name}</span></div>
                </span></div>
                <div className='event-direction' title='Bad Soden(Taunus)'><i className='icon'>arrow_forward</i>Bad
                    Soden(Taunus)</div>
                <div className='event-track'></div>
            </div>
        );
    }

    return divs;
}

export const StationEvent: React.FC<{ 'translation': Translations, 'station': (Station | Address), 'stationEventTrigger': boolean, 'setSubOverlayHidden': React.Dispatch<React.SetStateAction<boolean>>, 'setStationEventTrigger': React.Dispatch<React.SetStateAction<boolean>> }> = (props) => {

    const [eventStations, setEventStations] = useState<StationEvents>({ station: {id: '', name: ''}, events: []});
    console.log(eventStations);

    let byScheduleTime = true;
    let direction = 'BOTH';
    let eventCount = 20;
    let stationID = (props.station as Station).id;
    let time = 1646226000;

    useEffect(() => {
        if (props.stationEventTrigger) {
            let requestURL = 'https://europe.motis-project.de/?elm=StationEvents';
            fetch(requestURL, getStationEvent(byScheduleTime, direction, eventCount, stationID, time))
                .then(res => res.json())
                .then((res: RailVizStationRequest) => {
                    console.log('StationEvents brrrrr');
                    console.log(res);
                    setEventStations(res.content);
                });
        }
    }, [props.stationEventTrigger]);

    return (
        <div className='station-events'>
            <div className='header'>
                <div className='back' onClick={() => { props.setSubOverlayHidden(true); props.setStationEventTrigger(false) }}><i className='icon'>arrow_back</i></div>
                <div className='station'>{props.station.name}</div>
                <div className='event-type-picker'>
                    <div><input type='radio' id='station-departures' name='station-event-types' />
                        <label htmlFor='station-departures'>{props.translation.search.departure}</label></div>
                    <div><input type='radio' id='station-arrivals' name='station-event-types' />
                        <label htmlFor='station-arrivals'>{props.translation.search.arrival}</label></div>
                </div>
            </div>
            <div className='events'>
                <div className=''>
                    <div className='extend-search-interval search-before'><a>{props.translation.connections.extendBefore}</a></div>
                    <div className='event-list'>
                        <div className='date-header divider'><span>2.3.2022</span></div>
                        {(eventStations.station.id === '') ? <></> : stationEventDivGenerator(eventStations)}
                    </div>
                    <div className='divider footer'></div>
                    <div className='extend-search-interval search-after'><a>{props.translation.connections.extendAfter}</a></div>
                </div>
            </div>
        </div>
    );
}