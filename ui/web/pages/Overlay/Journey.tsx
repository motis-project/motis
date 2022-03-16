import React, { useEffect, useState } from 'react';

import { Transport, TransportInfo, Connection, Stop, TripId, FootRouting, Station, Trip } from '../Types/ConnectionTypes';
import { getFromLocalStorage, ModeLocalStorage } from '../App/LocalStorage';
import { Address } from '../Types/SuggestionTypes';
import { Translations } from '../App/Localization';
import { classToId } from './ConnectionRender';

import moment from 'moment';

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


const stopGenerator = (stops: Stop[]) => {
    let stopDivs = [];
    for (let index = 1; index < stops.length - 1; index++) {
        stopDivs.push(
            <div className='stop past' key={index}>
                <div className='timeline train-color-border bg'></div>
                <div className='timeline train-color-border progress' style={{ height: '100%' }}></div>
                <div className='time'>
                    <span className='past'>{moment.unix(stops[index].arrival.time).format('HH:mm')}</span>
                </div>
                <div className='delay'></div>
                <div className='station'>
                    <span>{stops[index].station.name}</span>
                </div>
            </div>
        );
    }
    return stopDivs;
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
        if (transport.move_type === 'Walk') {
            let requestURL = 'https://europe.motis-project.de/?elm=FootRoutingRequest';
            return fetch(requestURL, getWalkTime(
                connection.stops[transport.move.range.from].station.pos.lat,
                connection.stops[transport.move.range.from].station.pos.lng,
                connection.stops[transport.move.range.to].station.pos.lat,
                connection.stops[transport.move.range.to].station.pos.lng, toModes.walk.search_profile.max_duration * 60, toModes.walk.search_profile.profile, false, true, true))
                .then(res => res.json())
                .then((res: FootRouting) => {
                    console.log('Foot Request successful');
                    console.log(res);
                    walks.push(res.content.routes[0].routes[0].duration);
                });
        }
    })
    Promise.all(promises).then(results => {
        setWalkTimes(walks);
        console.log("Promise.all")
        console.log(walks);
    })
}


const getIntermediateStopsCount = (transport: Transport) => {
    return transport.move.range.to - transport.move.range.from - 1;
}


export const JourneyRender: React.FC<{ 'translation': Translations, 'connection': Connection, 'setSubOverlayHidden': React.Dispatch<React.SetStateAction<Boolean>>, 'setTrainSelected': React.Dispatch<React.SetStateAction<TripId>>, 'detailViewHidden': Boolean}> = (props) => {

    const [isIntermediateStopsCollapsed, setIsIntermediateStopsCollapsed] = useState<Boolean>(true);

    const [start, setStart] = useState<Station | Address>(getFromLocalStorage("motis.routing.from_location"));

    const [destination, setDestination] = useState<Station | Address>(getFromLocalStorage("motis.routing.to_location"));

    const [toModes, setToModes] = useState<ModeLocalStorage>(getFromLocalStorage('motis.routing.from_modes'));

    const [walkTimes, setWalkTimes] = useState<number[]>([]);

    console.log(props.detailViewHidden);

    useEffect(() => {
        if (props.connection.transports.length !== props.connection.trips.length) {
            fetchFoot(props.connection, toModes, setWalkTimes)
        }
    }, [props.connection]);

    let numberIntermediateStops = (props.connection.transports[0].move as TransportInfo).range.to - (props.connection.transports[0].move as TransportInfo).range.from;

    return (
        <>
            {props.connection.transports.length === 1 ?
                (props.connection.transports[0].move_type !== 'Walk') ?
                    <div className={'train-detail train-class-' + (props.connection.transports[0].move as TransportInfo).clasz}>
                        <div className='top-border'></div>
                        <div>
                            <div className={'train-box train-class-' + (props.connection.transports[0].move as TransportInfo).clasz + ' with-tooltip'}
                                data-tooltip={
                                    props.translation.connections.provider +
                                    ': ' + (props.connection.transports[0].move as TransportInfo).provider +
                                    '\n' +
                                    ((true) ?
                                        props.translation.connections.trainNr +
                                        ': ' +
                                        props.connection.trips[0].id.train_nr
                                        :
                                        props.translation.connections.lineId +
                                        ': ' +
                                        props.connection.trips[0].id.line_id)}
                                onClick={() => { props.setSubOverlayHidden(false); props.setTrainSelected(props.connection.trips[0].id); }}>
                                <svg className='train-icon'>
                                    <use xlinkHref={classToId((props.connection.transports[0].move as TransportInfo).clasz)}></use>
                                </svg>
                                <span className='train-name'>{(props.connection.transports[0].move as TransportInfo).name}</span>
                            </div>
                        </div>
                        <div className='first-stop'>
                            <div className='stop past'>
                                <div className='timeline train-color-border'></div>
                                <div className='time'>
                                    <span className='past'>{moment.unix(props.connection.stops[0].departure.time).format('HH:mm')}</span>
                                </div>
                                <div className='delay'></div>
                                <div className='station'>
                                    <span>{props.connection.stops[0].station.name}</span>
                                </div>
                            </div>
                        </div>
                        <div className='direction past'>
                            <div className='timeline train-color-border'></div>
                            <i className='icon'>arrow_forward</i>
                            {(props.connection.transports[0].move as TransportInfo).direction}
                        </div>
                        <div className={
                            (numberIntermediateStops === 1) ?
                                'intermediate-stops-toggle past' :
                                'intermediate-stops-toggle clickable past'}
                            onClick={() => setIsIntermediateStopsCollapsed(!isIntermediateStopsCollapsed)}>
                            <div className='timeline-container'>
                                <div className='timeline train-color-border bg'></div>
                                <div className='timeline train-color-border progress' style={{ height: '100%' }}></div>
                            </div>
                            {(numberIntermediateStops === 1) ?
                                <></> :
                                <div className='expand-icon'>
                                    <i className='icon'>expand_less</i>
                                    <i className='icon'>expand_more</i>
                                </div>
                            }
                            <span>
                                {props.translation.connections.tripIntermediateStops(numberIntermediateStops - 1) + ' (' + duration(props.connection.stops[0].departure.time, props.connection.stops[props.connection.stops.length - 1].arrival.time) + ')'}
                            </span>
                        </div>
                        <div className={isIntermediateStopsCollapsed ? 'intermediate-stops collapsed' : 'intermediate-stops expanded'}>
                            {stopGenerator(props.connection.stops)}
                        </div>
                        <div className='last-stop'>
                            <div className='stop past'>
                                <div className='timeline train-color-border'></div>
                                <div className='time'>
                                    <span className='past'>{moment.unix(props.connection.stops[props.connection.stops.length - 1].arrival.time).format('HH:mm')}</span>
                                </div>
                                <div className='delay'></div>
                                <div className='station'>
                                    <span>{props.connection.stops[props.connection.stops.length - 1].station.name}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    :
                    <>
                    <div className="train-detail train-class-walk initial-walk">
                        <div className="top-border"></div>
                        <div>
                            <div className="train-box train-class-walk">
                                <svg className="train-icon">
                                    <use xlinkHref="#walk"></use>
                                </svg>
                            </div>
                        </div>
                        <div className="first-stop">
                            <div className="stop past">
                                <div className="timeline train-color-border"></div>
                                <div className="time">
                                    <span className="past">{moment.unix(props.connection.stops[0].departure.time).format('HH:mm')}</span>
                                </div>
                                <div className="delay"></div>
                                <div className="station">
                                    <span className="virtual">{start.name}</span>
                                </div>
                            </div>
                        </div>
                        <div className="intermediate-stops-toggle">
                            <div className="timeline-container">
                                <div className="timeline train-color-border bg"></div>
                                <div className="timeline train-color-border progress" style={{ height: '100%' }}></div>
                            </div>
                            <div className="expand-icon"></div>
                            <span>{props.translation.connections.tripWalk(duration(props.connection.stops[0].departure.time, props.connection.stops[1].arrival.time))}</span>
                        </div>
                        <div className="last-stop">
                            <div className="stop past">
                                <div className="timeline train-color-border"></div>
                                <div className="time">
                                    <span className="past">{moment.unix(props.connection.stops[props.connection.stops.length - 1].arrival.time).format('HH:mm')}</span>
                                </div>
                                <div className="delay"></div>
                                <div className="station">
                                    <span className="virtual">{destination.name}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    </>
                :
                <TransportDivs
                    connection={props.connection}
                    isCollapsed={isIntermediateStopsCollapsed}
                    collapseSetter={setIsIntermediateStopsCollapsed}
                    setSubOverlayHidden={props.setSubOverlayHidden}
                    setTrainSelected={props.setTrainSelected}
                    walkTimes={walkTimes}
                    translation={props.translation}
                />
            }
        </>
    );
};


interface JourneyElem {
    hasWalk: boolean,
    walkTime: number,
    transport: Transport,
    stops: Stop[],
    trip: Trip
}

const TransportDivs: React.FC<{ 'connection': Connection, 'isCollapsed': Boolean, 'collapseSetter': React.Dispatch<React.SetStateAction<Boolean>>, 'setSubOverlayHidden': React.Dispatch<React.SetStateAction<Boolean>>, 'setTrainSelected': React.Dispatch<React.SetStateAction<TripId>>, 'walkTimes': number[], 'translation': Translations }> = (props) => {

    const [transports, setTransports] = useState<JourneyElem[]>([]);

    useEffect(() => {
        let t: JourneyElem[] = []
        let hasWalk = false;
        let walkCounter = 0;
        props.connection.transports.map((transport: Transport, index) => {
            if (isTransportInfo(transport) && hasWalk) {
                t.push({hasWalk: true, walkTime: props.walkTimes[walkCounter], transport: transport, stops: props.connection.stops, trip: props.connection.trips[index - walkCounter]});
                hasWalk = false;
                walkCounter += 1;
            } else if (isTransportInfo(transport)) {
                t.push({hasWalk: false, walkTime: 0, transport: transport, stops: props.connection.stops, trip: props.connection.trips[index - walkCounter]});
            } else {
                hasWalk = true;
            }
        })
        setTransports(t);
    }, [props.walkTimes]);

    console.log('WalkTimes');
    console.log(props.walkTimes);

    return (
        <>
            {props.walkTimes.length === 0 ?
                <></>
                :
                <>
                    {transports.map((transport: JourneyElem, index) => (
                        <div className={'train-detail train-class-' + (transport.transport.move as TransportInfo).clasz} key={index}>
                            <div className='top-border'></div>
                            <div>
                                <div className={'train-box train-class-' + (transport.transport.move as TransportInfo).clasz + ' with-tooltip'}
                                    data-tooltip={
                                        props.translation.connections.provider +
                                        ': ' + (transport.transport.move as TransportInfo).provider +
                                        '\n' +
                                        ((true) ?
                                            props.translation.connections.trainNr +
                                            ': ' +
                                            props.connection.trips[0].id.train_nr
                                            :
                                            props.translation.connections.lineId +
                                            ': ' +
                                            props.connection.trips[0].id.line_id)}
                                    onClick={() => { props.setSubOverlayHidden(false); props.setTrainSelected(transport.trip.id) }}>
                                    <svg className='train-icon' onClick={() => { props.setSubOverlayHidden(false); props.setTrainSelected(transport.trip.id) }}>
                                        <use xlinkHref={classToId((transport.transport.move as TransportInfo).clasz)}></use>
                                    </svg>
                                    <span className='train-name'>{(transport.transport.move as TransportInfo).name}</span>
                                </div>
                            </div>
                            {index === 0 ?
                                <></>
                                :
                                <div className='train-top-line'>
                                    <span>{transport.hasWalk ? transport.walkTime + 'min Fußweg' : duration(transport.stops[(transport.transport.move as TransportInfo).range.from].arrival.time, transport.stops[(transport.transport.move as TransportInfo).range.from].departure.time,) + ' Umstieg'}</span>
                                </div>
                            }
                            <div className='first-stop'>
                                <div className='stop past'>
                                    <div className='timeline train-color-border'></div>
                                    <div className='time'>
                                        <span className='past'>{moment.unix(transport.stops[(transport.transport.move as TransportInfo).range.from].departure.time).format('HH:mm')}</span>
                                    </div>
                                    <div className='delay'></div>
                                    <div className='station'>{transport.stops[(transport.transport.move as TransportInfo).range.from].station.name}</div>
                                </div>
                            </div>
                            <div className='intermediate-stops-toggle clickable past' onClick={() => props.collapseSetter(!props.isCollapsed)}>
                                <div className='timeline-container'>
                                    <div className='timeline train-color-border bg'></div>
                                    <div className='timeline train-color-border progress' style={{ height: '100%' }}></div>
                                </div>
                                <div className='expand-icon'>
                                    <i className='icon'>expand_less</i>
                                    <i className='icon'>expand_more</i>
                                </div>
                                <span>{props.translation.connections.tripIntermediateStops(getIntermediateStopsCount(transport.transport)) + ' (' + duration(props.connection.stops[(transport.transport.move as TransportInfo).range.from].departure.time, props.connection.stops[(transport.transport.move as TransportInfo).range.to].arrival.time) + ')'}</span>
                            </div>
                            <div className={props.isCollapsed ? 'intermediate-stops collapsed' : 'intermediate-stops expanded'}>
                                {stopGenerator(transport.stops)}
                            </div>
                            <div className="last-stop">
                                <div className="stop past">
                                    <div className="timeline train-color-border"></div>
                                    <div className="time">
                                        <span className="past">{moment.unix(transport.stops[(transport.transport.move as TransportInfo).range.to].arrival.time).format('HH:mm')}</span>
                                    </div>
                                    <div className="delay"></div>
                                    <div className="station">
                                        <span className="virtual">{transport.stops[(transport.transport.move as TransportInfo).range.to].station.name}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    ))}
                </>
            }
        </>
    )
}