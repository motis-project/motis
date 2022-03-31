import React, { useEffect, useState } from 'react';

import moment from 'moment';

import { Transport, TransportInfo, Connection, Stop, TripId, FootRouting, Station, Trip, WalkInfo } from '../Types/Connection';
import { getFromLocalStorage, ModeLocalStorage } from '../App/LocalStorage';
import { Address } from '../Types/SuggestionTypes';
import { Translations } from '../App/Localization';
import { classToId, getClasz } from './ConnectionRender';
import { SubOverlayEvent } from '../Types/EventHistory';


interface Journey {
    'translation': Translations,
    'connection': Connection,
    'setTrainSelected': React.Dispatch<React.SetStateAction<TripId>>,
    'subOverlayContent': SubOverlayEvent[],
    'setSubOverlayContent': React.Dispatch<React.SetStateAction<SubOverlayEvent[]>>,
}


const isTransportInfo = (transport: Transport) => {
    return transport.move_type === 'Transport';
}


export const duration = (start: number, dest: number) => {
    let difference = moment.unix(dest).diff(moment.unix(start), 'minutes')
    let hours = Math.floor(difference / 60)
    let minutes = difference % 60
    let returnString = (hours > 0) ? hours + 'h ' + minutes + 'min' : minutes + 'min'

    return returnString
}


const getWalkTime = (latStart: number, lngStart: number, latDest: number, lngDest: number, durationLimit: number, profile: string, includeEdges: boolean, includePath: boolean, includeSteps: boolean) => {
    return {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            destination: { type: "Module", target: "/ppr/route" },
            content_type: 'FootRoutingRequest',
            content: { start: { lat: latStart, lng: lngStart }, destinations: [{ lat: latDest, lng: lngDest }], search_options: { duration_limit: durationLimit, profile: profile }, include_edges: includeEdges, include_path: includePath, include_steps: includeSteps }
        })
    }
}


const fetchFoot = async (connection: Connection, toModes: ModeLocalStorage, setWalkTimes: React.Dispatch<React.SetStateAction<number[]>>) => {
    let walks = [];
    const promises = connection.transports.map((transport: Transport) => {
        if (transport.move_type === 'Walk' && getClasz(transport) === 'walk') {
            let requestURL = 'https://europe.motis-project.de/?elm=FootRoutingRequest';
            return fetch(requestURL, getWalkTime(
                connection.stops[transport.move.range.from].station.pos.lat,
                connection.stops[transport.move.range.from].station.pos.lng,
                connection.stops[transport.move.range.to].station.pos.lat,
                connection.stops[transport.move.range.to].station.pos.lng, toModes.walk.search_profile.max_duration * 60, toModes.walk.search_profile.profile, false, true, true))
                .then(res => res.json())
                .then((res: FootRouting) => {
                    walks.push(res.content.routes[0].routes[0].duration);
                });
        }
    })
    Promise.all(promises).then(results => {
        setWalkTimes(walks);
    })
}


const getIntermediateStopsCount = (transport: Transport) => {
    return transport.move.range.to - transport.move.range.from - 1;
}

const getMumoString = (mumoType: string, translation: Translations) => {
    switch (mumoType) {
        case 'walk' || 'foot':
            return translation.connections.walk;
        case 'bike':
            return translation.connections.bike;
        case 'car':
            return translation.connections.car;
        default:
            return undefined;
    }
}

interface JourneyElem {
    hasWalk: boolean,
    walkTime: number,
    transport: Transport,
    stops: Stop[],
    stopsToRender?: Stop[],
    trip?: Trip,
    walkInfo: boolean,
    expandString?: string,
    index: number
}


export const JourneyRender: React.FC<Journey> = (props) => {

    const [start, setStart] = useState<Station | Address>(getFromLocalStorage("motis.routing.from_location"));

    const [destination, setDestination] = useState<Station | Address>(getFromLocalStorage("motis.routing.to_location"));

    const [toModes, setToModes] = useState<ModeLocalStorage>(getFromLocalStorage('motis.routing.from_modes'));

    const [walkTimes, setWalkTimes] = useState<number[]>([]);
    const [transports, setTransports] = useState<JourneyElem[]>([]);

    useEffect(() => {
        let t: JourneyElem[] = []
        let hasWalk = false;
        let walkCounter = 0;
        let tripCounter = 0;
        props.connection.transports.map((transport: Transport, index) => {
            if (isTransportInfo(transport) && hasWalk) {
                t.push({ hasWalk: true, walkTime: walkTimes[walkCounter], transport: transport, stops: props.connection.stops, stopsToRender: props.connection.stops.slice(transport.move.range.from + 1, transport.move.range.to), trip: props.connection.trips[tripCounter], walkInfo: false, index: index });
                hasWalk = false;
                walkCounter += 1;
                tripCounter += 1;
            } else if (isTransportInfo(transport)) {
                t.push({ hasWalk: false, walkTime: 0, transport: transport, stops: props.connection.stops, stopsToRender: props.connection.stops.slice(transport.move.range.from + 1, transport.move.range.to), trip: props.connection.trips[tripCounter], walkInfo: false, index: index });
                tripCounter += 1;
            } else if (!isTransportInfo(transport) && (index == 0 || index == props.connection.transports.length - 1)) {
                t.push({ hasWalk: false, walkTime: 0, transport: transport, stops: props.connection.stops, stopsToRender: props.connection.stops.slice(transport.move.range.from + 1, transport.move.range.to), walkInfo: true, expandString: getMumoString(getClasz(transport).toString(), props.translation).toString(), index: index });
            } else {
                hasWalk = true;
            }
        })
        setTransports(t);
    }, [walkTimes]);

    useEffect(() => {
        if ((props.connection.transports.length !== props.connection.trips.length)
            ||
            (props.connection.transports[0].move_type === 'Walk' && (props.connection.transports[0].move as WalkInfo).mumo_type === 'foot')) {
            fetchFoot(props.connection, toModes, setWalkTimes)
        } else {
            setWalkTimes([-1]);
        }
    }, [props.connection]);

    return (
        <>
            {transports.map((transport: JourneyElem, index) => (
                <div className={`train-detail train-class-${getClasz(transport.transport)} ${(transport.walkInfo) ? 'initial-walk' : ''}`} key={index}>
                    <div className='top-border'></div>
                    <div>
                        <div className={`train-box train-class-${getClasz(transport.transport)} ${(transport.walkInfo) ? '' : 'with-tooltip'}`}
                            data-tooltip={`${(transport.walkInfo) ?
                                '' :
                                `${props.translation.connections.provider}: ${(transport.transport.move as TransportInfo).provider}\n${props.translation.connections.trainNr}: ${(transport.trip !== undefined) ? transport.trip.id.train_nr : ''}`}`}
                            onClick={() => {
                                    props.setTrainSelected(transport.trip.id);
                                    props.setSubOverlayContent([...props.subOverlayContent, {id: 'tripView', train: transport.trip.id}]);
                                }}>
                            <svg className='train-icon'>
                                <use xlinkHref={classToId(transport.transport)}></use>
                            </svg>
                            {(transport.walkInfo) ? <></> : <span className='train-name'>{(transport.transport.move as TransportInfo).name}</span>}
                        </div>
                    </div>
                    {(index === 0 || (index > 0 && transports[index - 1].walkInfo)) ?
                        <></>
                        :
                        <div className='train-top-line'>
                            <span>{(transport.hasWalk) ? `${props.translation.connections.walkDuration(`${transport.walkTime}min`)}` : props.translation.connections.interchangeDuration(`${duration(transport.stops[(transport.transport.move as TransportInfo).range.from].arrival.time, transport.stops[(transport.transport.move as TransportInfo).range.from].departure.time)}`)}</span>
                        </div>
                    }
                    <div className='first-stop'>
                        <div className='stop future'>
                            <div className='timeline train-color-border'></div>
                            <div className='time'>
                                <span className='future'>{moment.unix(transport.stops[(transport.transport.move as TransportInfo).range.from].departure.time).format('HH:mm')}</span>
                            </div>
                            <div className='delay'></div>
                            <div    className='station'
                                    onClick={() => {
                                        props.setSubOverlayContent([...props.subOverlayContent, {id: 'stationEvent', station: (transport.stops[transport.transport.move.range.from].station.name === 'START') ? start : transport.stops[transport.transport.move.range.from].station, stationTime: moment.unix(transport.stops[(transport.transport.move as TransportInfo).range.from].departure.time)}]);
                                    }}>
                                <span>
                                    {(transport.stops[transport.transport.move.range.from].station.name === 'START') ? (start as Station).name : transport.stops[transport.transport.move.range.from].station.name}
                                </span>
                            </div>
                        </div>
                    </div>
                    {transport.walkInfo ?
                        <></>
                        :
                        <div className="direction future">
                            <div className="timeline train-color-border"></div>
                            <i className="icon">arrow_forward</i>
                            {(transport.transport.move as TransportInfo).direction}
                        </div>
                    }
                    <IntermediateStops  transport={transport}
                                        connection={props.connection}
                                        translation={props.translation}
                                        subOverlayContent={props.subOverlayContent}
                                        setSubOverlayContent={props.setSubOverlayContent}/>
                    <div className="last-stop">
                        <div className="stop future">
                            <div className="timeline train-color-border"></div>
                            <div className="time">
                                <span className="future">{moment.unix(transport.stops[(transport.transport.move as TransportInfo).range.to].arrival.time).format('HH:mm')}</span>
                            </div>
                            <div className="delay"></div>
                            <div    className="station"
                                    onClick={() => {
                                        props.setSubOverlayContent([...props.subOverlayContent, {id: 'stationEvent', station: (transport.stops[transport.transport.move.range.to].station.name === 'END') ? destination : transport.stops[transport.transport.move.range.to].station, stationTime: moment.unix(transport.stops[(transport.transport.move as TransportInfo).range.to].arrival.time)}]);
                                    }}>
                                <span>
                                    {(transport.stops[transport.transport.move.range.to].station.name === 'END') ? (destination as Station).name : transport.stops[transport.transport.move.range.to].station.name}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            ))}
        </>
    );
};

const IntermediateStops: React.FC<{'transport': JourneyElem, 'connection': Connection, 'translation': Translations, 'subOverlayContent': SubOverlayEvent[], 'setSubOverlayContent': React.Dispatch<React.SetStateAction<SubOverlayEvent[]>>}> = (props) => {

    const [isIntermediateStopsCollapsed, setIsIntermediateStopsCollapsed] = useState<boolean>(props.subOverlayContent.length === 0);

    return (
        <>
            <div className={`intermediate-stops-toggle ${(getIntermediateStopsCount(props.transport.transport) > 0) ? 'clickable' : ''} future`} onClick={() => setIsIntermediateStopsCollapsed(!isIntermediateStopsCollapsed)}>
                <div className='timeline-container'>
                    <div className='timeline train-color-border bg'></div>
                    <div className='timeline train-color-border progress' style={{ height: '0%' }}></div>
                </div>
                {(props.transport.walkInfo || getIntermediateStopsCount(props.transport.transport) === 0) ?
                    <div className='expand-icon'></div> 
                    :
                    isIntermediateStopsCollapsed ?
                    <div className='expand-icon'>
                        <i className='icon'>expand_less</i>
                        <i className='icon'>expand_more</i>
                    </div>
                    :
                    <div className='expand-icon'>
                        <i className='icon'>expand_more</i>
                        <i className='icon'>expand_less</i>
                    </div>
                }
                <span>{`${(props.transport.walkInfo) ? props.transport.expandString : props.translation.connections.tripIntermediateStops(getIntermediateStopsCount(props.transport.transport))} (${duration(props.connection.stops[(props.transport.transport.move as TransportInfo).range.from].departure.time, props.connection.stops[(props.transport.transport.move as TransportInfo).range.to].arrival.time)})`}</span>
            </div>
            <div className={isIntermediateStopsCollapsed ? 'intermediate-stops collapsed' : 'intermediate-stops expanded'}>
                {props.transport.stopsToRender.map((stop: Stop, index) => (
                    <div    className='stop future' 
                            key={index}
                            onClick={() => {
                                props.setSubOverlayContent([...props.subOverlayContent, {id: 'stationEvent', station: stop.station, stationTime: moment.unix(stop.departure.time)}]);
                            }}>
                        <div className='timeline train-color-border bg'></div>
                        <div className='timeline train-color-border progress' style={{ height: '0%' }}></div>
                        {props.subOverlayContent.length !== 0 ?
                            <div className='time'>
                                <div className="arrival">
                                    <span className="future">{moment.unix(stop.arrival.time).format('HH:mm')}</span>
                                </div>
                                <div className="departure">
                                    <span className="future">{moment.unix(stop.departure.time).format('HH:mm')}</span>
                                </div> 
                            </div>
                            :
                            <div className='time'>
                                <span className='future'>{moment.unix(stop.departure.time).format('HH:mm')}</span>
                            </div>
                        }
                        <div className='delay'></div>
                        <div className='station'>
                            <span>{stop.station.name}</span>
                        </div>
                    </div>
                ))}
            </div>
        </>
    );
};