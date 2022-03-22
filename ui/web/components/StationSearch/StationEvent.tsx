import React, { useEffect, useState } from 'react';

import moment from 'moment';
import { Events, RailVizStationRequest, StationEvents } from '../Types/RailvizStationEvent';
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

const stationEventDivGenerator = (eventsToDisplay: Events[], translation: Translations, displayDirection: string, direction: string) => {

    let divs = [];
    for (let index = 0; index < eventsToDisplay.length; index++) {
        if (eventsToDisplay[index].type === displayDirection || eventsToDisplay[index].type === '') {
            (eventsToDisplay[index].dummyEvent) ?
                divs.push(
                    <div className="date-header divider" key={index}>
                        <span>{eventsToDisplay[index].dummyEvent}</span>
                    </div>
                ):
                divs.push(
                    <div className='station-event' key={index}>
                        <div className='event-time'>{moment.unix(eventsToDisplay[index].event.time).format('HH:mm')}</div>
                        <div className='event-train'><span>
                            <div className={'train-box train-class-' + eventsToDisplay[index].trips[0].transport.clasz + ' with-tooltip'} data-tooltip={translation.connections.provider + ': ' + eventsToDisplay[index].trips[0].transport.provider + '\n' + translation.connections.trainNr + ': ' + eventsToDisplay[index].trips[0].transport.train_nr}><svg className='train-icon'>
                                <use xlinkHref={classToId(eventsToDisplay[index].trips[0].transport.clasz)}></use>
                            </svg><span className='train-name'>{eventsToDisplay[index].trips[0].transport.name}</span></div>
                        </span></div>
                        <div className='event-direction' title={eventsToDisplay[index].trips[0].transport.direction}><i className='icon'>arrow_forward</i>{eventsToDisplay[index].trips[0].transport.direction}</div>
                        <div className='event-track'></div>
                    </div>
                );
        }
    }  
    console.log(divs[divs.length-1].props.className);

    return divs;
}


const onClickHandler = (byScheduleTime: boolean, direction: string, eventCount: number, stationID: string, time: number, eventStations: Events[], setEventStations: React.Dispatch<React.SetStateAction<Events[]>>, setMinTime: React.Dispatch<React.SetStateAction<number>>, setMaxTime: React.Dispatch<React.SetStateAction<number>>, setEventsToDisplay: React.Dispatch<React.SetStateAction<Events[]>>, translation: Translations) => {
    let requestURL = 'https://europe.motis-project.de/?elm=StationEvents';
    fetch(requestURL, getStationEvent(byScheduleTime, direction, eventCount, stationID, time))
        .then(res => res.json())
        .then((res: RailVizStationRequest) => {
            console.log('StationEvents brrrrr');
            console.log(res);
            if (direction === 'EARLIER') {
                insertDateHeader(setEventsToDisplay, [...res.content.events, ...eventStations], translation);
                setEventStations([...res.content.events, ...eventStations]);
                setMinTime(Math.floor(res.content.events[0].event.time / 1000) * 1000);
            } else {
                insertDateHeader(setEventsToDisplay, [...eventStations, ...res.content.events], translation);
                setEventStations([...eventStations, ...res.content.events]);
                setMaxTime(Math.floor(res.content.events[res.content.events.length - 1].event.time / 1000) * 1000);
            }
        });
}


const insertDateHeader = (setEventsToDisplay: React.Dispatch<React.SetStateAction<Events[]>>, events: Events[], translation: Translations) => {
    let dummyIndexes = [0];
    let eventsWithDummies = [...events];
    let previousConnectionDay = moment.unix(events[0].event.time);
    let dummyDays = [previousConnectionDay.format(translation.dateFormat)];

    for (let i = 1; i < events.length; i++) {
        if ( moment.unix(events[i].event.time).day() !== previousConnectionDay.day()) {
            dummyIndexes.push(i);
            previousConnectionDay.add(1, 'day');
            dummyDays.push(previousConnectionDay.format(translation.dateFormat));
        }
    };
    dummyIndexes.map((val, idx) => {
        eventsWithDummies.splice(val + idx, 0, dummyEvent(dummyDays[idx]));
    })
    setEventsToDisplay(eventsWithDummies);
};

const dummyEvent = (time: string): Events => {
    return { trips: [], type: '', event: { time: 0, schedule_time: 0, track: '', schedule_track: '', valid: false, reason: '' }, dummyEvent: time }
}

export const StationEvent: React.FC<{ 'translation': Translations, 'station': (Station | Address), 'stationEventTrigger': boolean, 'setSubOverlayHidden': React.Dispatch<React.SetStateAction<boolean>>, 'setStationEventTrigger': React.Dispatch<React.SetStateAction<boolean>>, 'searchDate': moment.Moment }> = (props) => {

    const [eventStations, setEventStations] = useState<Events[]>(null);

    const [eventsToDisplay, setEventsToDisplay] = useState<Events[]>(null);

    let byScheduleTime = true;
    let eventCount = 20;
    let stationID = (props.station as Station).id;
    const [time, setTime] = useState<number>((props.searchDate === null) ? 0 : props.searchDate.unix());
    const [minTime, setMinTime] = useState<number>(0);
    const [maxTime, setMaxTime] = useState<number>(Number.MAX_VALUE);
    const [displayDirection, setDisplayDirection] = useState<string>('DEP');
    const [direction, setDirection] = useState<string>('BOTH');

    useEffect(() => {
        if (props.stationEventTrigger && stationID !== '') {
            let requestURL = 'https://europe.motis-project.de/?elm=StationEvents';
            fetch(requestURL, getStationEvent(byScheduleTime, direction, eventCount, stationID, time))
                .then(res => res.json())
                .then((res: RailVizStationRequest) => {
                    console.log('StationEvents brrrrr');
                    console.log(res);
                    insertDateHeader(setEventsToDisplay, res.content.events, props.translation);
                    setEventStations(res.content.events);
                    setMinTime(res.content.events[0].event.time);
                    setMaxTime(res.content.events[res.content.events.length - 1].event.time);
                });
        }
    }, [props.stationEventTrigger, direction, props.station]);

    return (
        <div className='station-events'>
            <div className='header'>
                <div className='back' onClick={() => { props.setSubOverlayHidden(true); props.setStationEventTrigger(false) }}><i className='icon'>arrow_back</i></div>
                <div className='station'>{props.station.name}</div>
                <div className='event-type-picker'>
                    <div>
                        <input type='radio' id='station-departures' name='station-event-types' onClick={() => { setDisplayDirection('DEP') }} />
                        <label htmlFor='station-departures'>{props.translation.search.departure}</label>
                    </div>
                    <div>
                        <input type='radio' id='station-arrivals' name='station-event-types' onClick={() => { setDisplayDirection('ARR') }} />
                        <label htmlFor='station-arrivals'>{props.translation.search.arrival}</label>
                    </div>
                </div>
            </div>
            <div className='events'>
                <div className=''>
                    <div className='extend-search-interval search-before' onClick={() => { onClickHandler(byScheduleTime, 'EARLIER', eventCount, stationID, minTime, eventStations, setEventStations, setMinTime, setMaxTime, setEventsToDisplay, props.translation) }}><a>{props.translation.connections.extendBefore}</a></div>
                    <div className='event-list'>
                        {(eventsToDisplay) ? stationEventDivGenerator(eventsToDisplay, props.translation, displayDirection, direction) : <></>}
                    </div>
                    <div className='divider footer'></div>
                    <div className='extend-search-interval search-after' onClick={() => { onClickHandler(byScheduleTime, 'LATER', eventCount, stationID, maxTime, eventStations, setEventStations, setMinTime, setMaxTime, setEventsToDisplay, props.translation) }}><a>{props.translation.connections.extendAfter}</a></div>
                </div>
            </div>
        </div>
    );
}