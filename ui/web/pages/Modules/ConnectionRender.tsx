import moment from 'moment';
import React, { useEffect, useState } from 'react';
import Index from '..';
import { Transport, TransportInfo, Connection, Stop, TripId, FootRouting, Station } from './ConnectionTypes';
import { Mode } from './IntermodalRoutingTypes';
import { getFromLocalStorage, ModeLocalStorage } from './LocalStorage';
import { Address } from './SuggestionTypes';

const isTransportInfo = (transport: Transport) => {
    return transport.move_type === 'Transport';
}

let arrLength = 0;

const isArrLengthOne = (connection: Transport[]) => {
    arrLength = connection.length;
    return connection.length === 1;
}

const transportForLoop = (connection: Transport[], setDetailViewHidden: React.Dispatch<React.SetStateAction<Boolean>>) => {
    var elements = [];
    var percentage = 0;
    var rangeMax = connection[connection.length - 1].move.range.to;
    var walk = 0;
    var prevLength = 0;
    for (let index = 0; index < connection.length; index++) {
        percentage = (connection[index].move.range.to - connection[index].move.range.from + walk) / rangeMax;
        isTransportInfo(connection[index]) ?
            elements.push(
                <g className={'part train-class-' + (connection[index].move as TransportInfo).clasz + ' acc-0'} key={index}>
                    <line x1={prevLength} y1='12' x2={(percentage * 326 + prevLength)} y2='12' className='train-line'></line>
                    <circle cx={prevLength + 4} cy='12' r='12' className='train-circle'></circle>
                    <use xlinkHref={classToId((connection[index].move as TransportInfo).clasz)} className='train-icon' x={prevLength - 4} y='4' width='16' height='16'></use>
                    <text x={prevLength - 6} y='40' textAnchor='start' className='train-name'>{(connection[index].move as TransportInfo).name}</text>
                    <rect x={prevLength} y='0' width={(percentage * 326 + prevLength)} height='24' className='tooltipTrigger' onClick={() => setDetailViewHidden(false)}></rect>
                </g>
            ) :
            walk = 1;
        if (isTransportInfo(connection[index])) {
            prevLength = prevLength + (percentage * 326);
        }
    }
    return elements;
}

const stopGenerator = (stops: Stop[]) => {
    let stopDivs = [];
    for (let index = 1; index < stops.length - 1; index++) {
        stopDivs.push(
            <div className='stop past' key={index}>
                <div className='timeline train-color-border bg'></div>
                <div className='timeline train-color-border progress' style={{ height: '100%' }}></div>
                <div className='time'>
                    <span className='past'>{displayTime(stops[index].arrival.time)}</span>
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


const classToId = (classz: Number) => {
    switch (classz) {
        case 0:
            return '#plane';
            break;
        case 1:
            return '#train';
            break;
        case 2:
            return '#train';
            break;
        case 3:
            return '#bus';
            break;
        case 4:
            return '#train';
            break;
        case 5:
            return '#train';
            break;
        case 6:
            return '#train';
            break;
        case 7:
            return '#sbahn';
            break;
        case 8:
            return '#ubahn';
            break;
        case 9:
            return '#tram';
            break;
        case 11:
            return '#ship';
            break;
        default:
            return '#bus';
            break;
    }
}


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


const fetchFoot = async(connection: Connection, toModes: ModeLocalStorage, setWalkTimes: React.Dispatch<React.SetStateAction<number[]>>) => {
    let walks = [];
    const promises = connection.transports.map((transport: Transport) => {
        if (transport.move_type === 'Walk'){
            let requestURL = 'https://europe.motis-project.de/?elm=FootRoutingRequest';
            return fetch(requestURL, getWalkTime(
                connection.stops[transport.move.range.from].station.pos.lat,
                connection.stops[transport.move.range.from].station.pos.lng,
                connection.stops[transport.move.range.to].station.pos.lat,
                connection.stops[transport.move.range.to].station.pos.lng, toModes.walk.search_profile.max_duration*60, toModes.walk.search_profile.profile, false, true, true))
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


export const ConnectionRender: React.FC<{ 'connection': Connection, 'setDetailViewHidden': React.Dispatch<React.SetStateAction<Boolean>> }> = (props) => {

    return (
        <svg width='335' height='40' viewBox='0 0 335 40'>
            <g>
                {isArrLengthOne(props.connection.transports) ?
                    isTransportInfo(props.connection.transports[0]) ?
                        <g className={'part train-class-' + (props.connection.transports[0].move as TransportInfo).clasz + ' acc-0'}>
                            <line x1='0' y1='12' x2='326' y2='12' className='train-line'></line>
                            <circle cx='4' cy='12' r='12' className='train-circle'></circle>
                            <use xlinkHref={classToId((props.connection.transports[0].move as TransportInfo).clasz)} className='train-icon' x='-4' y='4' width='16' height='16'></use>
                            <text x='-6' y='40' textAnchor='start' className='train-name'>{(props.connection.transports[0].move as TransportInfo).name}</text>
                            <rect x='0' y='0' width='323' height='24' className='tooltipTrigger'></rect>
                        </g>
                        :
                        <g className='part train-class-walk acc-0' onClick={() => console.log('test')/*fetchFoot(
                            props.connection.stops[props.connection.transports[0].move.range.from].station.pos.lat,
                            props.connection.stops[props.connection.transports[0].move.range.from].station.pos.lng,
                            props.connection.stops[props.connection.transports[0].move.range.to].station.pos.lat,
                            props.connection.stops[props.connection.transports[0].move.range.to].station.pos.lng,
                            900,
                            'default',
                            false,
                            true,
                            true,
                        props.setWalkTime)*/}>
                            <line x1='0' y1='12' x2='326' y2='12' className='train-line'></line>
                            <circle cx='4' cy='12' r='12' className='train-circle'></circle>
                            <use xlinkHref='#walk' className='train-icon' x='-4' y='4' width='16' height='16'></use>
                            <text x='0' y='40' textAnchor='start' className='train-name'></text>
                            <rect x='0' y='0' width='323' height='24' className='tooltipTrigger'></rect>
                        </g>
                    :
                    transportForLoop(props.connection.transports, props.setDetailViewHidden)
                }
            </g>
            <g className='destination'><circle cx='329' cy='12' r='6'></circle></g>
        </svg>
    );
};

export const JourneyRender: React.FC<{ 'connection': Connection, 'setSubOverlayHidden': React.Dispatch<React.SetStateAction<Boolean>>, 'setTrainSelected': React.Dispatch<React.SetStateAction<TripId>>, 'detailViewHidden': Boolean}> = (props) => {

    const [isIntermediateStopsCollapsed, setIsIntermediateStopsCollapsed] = useState<Boolean>(true);

    const [start, setStart] = useState<Station | Address>(getFromLocalStorage("motis.routing.from_location"));

    const [destination, setDestination] = useState<Station | Address>(getFromLocalStorage("motis.routing.to_location"));

    const [toModes, setToModes] = useState<ModeLocalStorage>(getFromLocalStorage('motis.routing.from_modes'));

    const [walkTimes, setWalkTimes] = useState<number[]>([]);

    console.log(props.detailViewHidden);

    useEffect(() => {
        if (props.connection.transports.length !== props.connection.trips.length){
            fetchFoot(props.connection, toModes, setWalkTimes)
            }
    }, [props.connection]);

    return (
            <>
                {isArrLengthOne(props.connection.transports) ?
                    (props.connection.transports[0].move_type !== 'Walk') ?
                        <div className={'train-detail train-class-' + (props.connection.transports[0].move as TransportInfo).clasz}>
                            <div className='top-border'></div>
                            <div>
                                <div className={'train-box train-class-' + (props.connection.transports[0].move as TransportInfo).clasz + ' with-tooltip'}
                                    data-tooltip={'Betreiber: DB Regio AG S-Bahn Rhein-Main \nZugnummer: ' + (props.connection.transports[0].move as TransportInfo).train_nr} onClick={() => { props.setSubOverlayHidden(false); props.setTrainSelected(props.connection.trips[0].id); }}>
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
                                        <span className='past'>{displayTime(props.connection.stops[0].departure.time)}</span>
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
                            <div className='intermediate-stops-toggle clickable past' onClick={() => setIsIntermediateStopsCollapsed(!isIntermediateStopsCollapsed)}>
                                <div className='timeline-container'>
                                    <div className='timeline train-color-border bg'></div>
                                    <div className='timeline train-color-border progress' style={{ height: '100%' }}></div>
                                </div>
                                <div className='expand-icon'>
                                    <i className='icon'>expand_less</i>
                                    <i className='icon'>expand_more</i>
                                </div>
                                <span>{((props.connection.transports[0].move as TransportInfo).range.to - (props.connection.transports[0].move as TransportInfo).range.from) === 1 ?
                                    'Fahrt ohne Zwischenhalt (' + displayDuration(new Date(props.connection.stops[props.connection.stops.length - 1].arrival.time).getTime() - new Date(props.connection.stops[0].departure.time).getTime()) + ')'
                                    :
                                    'Fahrt ' + ((props.connection.transports[0].move as TransportInfo).range.to - (props.connection.transports[0].move as TransportInfo).range.from) + ' Stationen (' + displayDuration(new Date(props.connection.stops[props.connection.stops.length - 1].arrival.time).getTime() - new Date(props.connection.stops[0].departure.time).getTime()) + ')'}</span>
                            </div>
                            <div className={isIntermediateStopsCollapsed ? 'intermediate-stops collapsed' : 'intermediate-stops expanded'}>
                                {stopGenerator(props.connection.stops)}
                            </div>
                            <div className='last-stop'>
                                <div className='stop past'>
                                    <div className='timeline train-color-border'></div>
                                    <div className='time'>
                                        <span className='past'>{displayTime(props.connection.stops[props.connection.stops.length - 1].arrival.time)}</span>
                                    </div>
                                    <div className='delay'></div>
                                    <div className='station'>
                                        <span>{props.connection.stops[props.connection.stops.length - 1].station.name}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        :
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
                                        <span className="past">{displayTime(props.connection.stops[0].departure.time)}</span>
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
                                <span>{'Fußweg (' + moment.unix(props.connection.stops[1].arrival.time).from(moment.unix(props.connection.stops[0].departure.time)) + ' min)'}</span>
                            </div>
                            <div className="last-stop">
                                <div className="stop past">
                                    <div className="timeline train-color-border"></div>
                                    <div className="time">
                                        <span className="past">{displayTime(props.connection.stops[props.connection.stops.length - 1].arrival.time)}</span>
                                    </div>
                                    <div className="delay"></div>
                                    <div className="station">
                                        <span className="virtual">{destination.name}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    :
                    <TransportDivs
                        connection={props.connection}
                        isCollapsed={isIntermediateStopsCollapsed}
                        collapseSetter={setIsIntermediateStopsCollapsed}
                        setSubOverlayHidden={props.setSubOverlayHidden}
                        setTrainSelected={props.setTrainSelected}
                        walkTimes={walkTimes}
                         />
                        }           
            </>
    );
};


interface temp {
    hasWalk: boolean,
    walkTime: number,
    transport: Transport,
}


const TransportDivs: React.FC<{'connection': Connection, 'isCollapsed': Boolean, 'collapseSetter': React.Dispatch<React.SetStateAction<Boolean>>, 'setSubOverlayHidden': React.Dispatch<React.SetStateAction<Boolean>>, 'setTrainSelected': React.Dispatch<React.SetStateAction<TripId>>, 'walkTimes': number[]}> = (props) => {

    const [transports, setTransports] = useState<temp[]>([]);

    useEffect(() => {
        let t: temp[] = []
        let hasWalk = false;
        let walkCounter = 0;
        props.connection.transports.map((transport: Transport, index) => {
            if (isTransportInfo(transport) && hasWalk) {
                t.push({hasWalk: true, walkTime: props.walkTimes[walkCounter], transport: transport});
                hasWalk = false;
                walkCounter += 1;
            } else if (isTransportInfo(transport)) {
                t.push({hasWalk: false, walkTime: 0, transport: transport});
            } else {
                hasWalk = true;
            }
        })
        setTransports(t);
    }, [props.walkTimes]);

    console.log('WalkTimes');
    console.log(props.walkTimes);

    return(
        <>
            {props.walkTimes.length === 0 ? 
                <></>
                :
                <>
                    {transports.map((transport: temp, index) => (
                        <div className={'train-detail train-class-' + (transport.transport.move as TransportInfo).clasz} key={index}>
                            <div className='top-border'></div>
                            <div>
                                <div className={'train-box train-class-' + (transport.transport.move as TransportInfo).clasz + ' with-tooltip'}
                                    data-tooltip={'Betreiber: DB Regio AG S-Bahn Rhein-Main \nZugnummer: ' + (transport.transport.move as TransportInfo).train_nr} onClick={() => { props.setSubOverlayHidden(false); props.setTrainSelected(props.connection.trips[index].id) }}>
                                    <svg className='train-icon' onClick={() => { props.setSubOverlayHidden(false); props.setTrainSelected(props.connection.trips[index].id) }}>
                                        <use xlinkHref={classToId((transport.transport.move as TransportInfo).clasz)}></use>
                                    </svg>
                                    <span className='train-name'>{(transport.transport.move as TransportInfo).name}</span>
                                </div>
                            </div>
                            {(index > 0 && transport.hasWalk) ?
                                <div className='train-top-line'>
                                    <span>{transport.walkTime + 'min Fußweg'}</span>
                                </div> :
                                (index === 0) ?
                                    <></> :
                                    <div>
                                        <span>{'min Umstieg'}</span>
                                    </div>
                            }
                            <div className='first-stop'>
                                <div className='stop past'>
                                    <div className='timeline train-color-border'></div>
                                    <div className='time'>
                                        <span className='past'>{moment.unix(props.connection.stops[(transport.transport.move as TransportInfo).range.from].departure.time).format('HH:mm')}</span>
                                    </div>
                                    <div className='delay'></div>
                                    <div className='station'>{props.connection.stops[(transport.transport.move as TransportInfo).range.from].station.name}</div>
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
                                <span>{((transport.transport.move as TransportInfo).range.to - (transport.transport.move as TransportInfo).range.from) === 1 ?
                                    'Fahrt ohne Zwischenhalt (' + displayDuration(new Date(props.connection.stops[(transport.transport.move as TransportInfo).range.to].arrival.time).getTime() - new Date(props.connection.stops[(transport.transport.move as TransportInfo).range.from].departure.time).getTime()) + ')'
                                    :
                                    'Fahrt ' + ((transport.transport.move as TransportInfo).range.to - (transport.transport.move as TransportInfo).range.from - 1) + ((((transport.transport.move as TransportInfo).range.to - (transport.transport.move as TransportInfo).range.from - 1) === 1) ? ' Station (' : ' Stationen (')
                                    + displayDuration(new Date(props.connection.stops[(transport.transport.move as TransportInfo).range.to].arrival.time).getTime()
                                        - new Date(props.connection.stops[(transport.transport.move as TransportInfo).range.from].departure.time).getTime()) + ')'}</span>
                            </div>
                            <div className={props.isCollapsed ? 'intermediate-stops collapsed' : 'intermediate-stops expanded'}>
                                {stopGenerator(props.connection.stops)}
                            </div>
                            <div className="last-stop">
                                <div className="stop past">
                                    <div className="timeline train-color-border"></div>
                                    <div className="time">
                                        <span className="past">{displayTime(props.connection.stops[(transport.transport.move as TransportInfo).range.to].arrival.time)}</span>
                                    </div>
                                    <div className="delay"></div>
                                    <div className="station">
                                        <span className="virtual">{props.connection.stops[(transport.transport.move as TransportInfo).range.to].station.name}</span>
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