import React, { useEffect, useState } from 'react';
import Index from '..';
import { Transport, TransportInfo, Connection, Stop, TripId, FootRouting} from './ConnectionTypes';

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
                <g className={'part train-class-' + (connection[index].move as TransportInfo).category_id + ' acc-0'} key={index}>
                    <line x1={prevLength} y1='12' x2={(percentage * 326 + prevLength)} y2='12' className='train-line'></line>
                    <circle cx={prevLength + 4} cy='12' r='12' className='train-circle'></circle>
                    <use xlinkHref={classToId((connection[0].move as TransportInfo).category_id)} className='train-icon' x={prevLength - 4} y='4' width='16' height='16'></use>
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
            content: { start: { lat: latStart, lng: lngStart }, destinations: [{ lat: latDest, lng: lngDest }, { lat: latDest, lng: lngDest }], search_options: { duration_limit: durationLimit, profile: profile }, include_edges: includeEdges, include_path: includePath, include_steps: includeSteps }
        })
    }
}

const transportDivs = (connection: Connection, isCollapsed: Boolean, collapseSetter: React.Dispatch<React.SetStateAction<Boolean>>, setSubOverlayHidden: React.Dispatch<React.SetStateAction<Boolean>>, setTrainSelected: React.Dispatch<React.SetStateAction<TripId>>, walkTime: number) => {
    let transDivs = [];
    for (let index = 0; index < connection.transports.length; index++) {
        if (isTransportInfo(connection.transports[index])) {
            transDivs.push(
                <div className={'train-detail train-class-' + (connection.transports[index].move as TransportInfo).category_id} key={index}>
                    <div className='top-border'></div>
                    <div>
                        <div className={'train-box train-class-' + (connection.transports[index].move as TransportInfo).category_id + ' with-tooltip'}
                            data-tooltip={'Betreiber: DB Regio AG S-Bahn Rhein-Main \nZugnummer: ' + (connection.transports[index].move as TransportInfo).train_nr} onClick={() => { setSubOverlayHidden(false); setTrainSelected(connection.trips[index].id) }}>
                            <svg className='train-icon' onClick={() => { setSubOverlayHidden(false); setTrainSelected(connection.trips[index].id) }}>
                                <use xlinkHref={classToId((connection.transports[index].move as TransportInfo).category_id)}></use>
                            </svg>
                            <span className='train-name'>{(connection.transports[index].move as TransportInfo).name}</span>
                        </div>
                    </div>
                    {(index !== 0) ?
                        <div className='train-top-line'>
                            <span>{walkTime + ' Fußweg'}</span>
                        </div> :
                        <></>}
                    <div className='first-stop'>
                        <div className='stop past'>
                            <div className='timeline train-color-border'></div>
                            <div className='time'>
                                <span className='past'>{displayTime(connection.stops[(connection.transports[index].move as TransportInfo).range.from].departure.time)}</span>
                            </div>
                            <div className='delay'></div>
                            <div className='station'>{connection.stops[index].station.name}</div>
                        </div>
                    </div>
                    <div className='intermediate-stops-toggle clickable past' onClick={() => collapseSetter(!isCollapsed)}>
                        <div className='timeline-container'>
                            <div className='timeline train-color-border bg'></div>
                            <div className='timeline train-color-border progress' style={{ height: '100%' }}></div>
                        </div>
                        <div className='expand-icon'>
                            <i className='icon'>expand_less</i>
                            <i className='icon'>expand_more</i>
                        </div>
                        <span>{((connection.transports[index].move as TransportInfo).range.to - (connection.transports[index].move as TransportInfo).range.from) === 1 ?
                            'Fahrt ohne Zwischenhalt (' + displayDuration(new Date(connection.stops[(connection.transports[index].move as TransportInfo).range.to].arrival.time).getTime() - new Date(connection.stops[(connection.transports[index].move as TransportInfo).range.from].departure.time).getTime()) + ')'
                            :
                            'Fahrt ' + ((connection.transports[index].move as TransportInfo).range.to - (connection.transports[index].move as TransportInfo).range.from - 1) + ((((connection.transports[index].move as TransportInfo).range.to - (connection.transports[index].move as TransportInfo).range.from - 1) === 1) ? ' Station (' : ' Stationen (')
                            + displayDuration(new Date(connection.stops[(connection.transports[index].move as TransportInfo).range.to].arrival.time).getTime()
                                - new Date(connection.stops[(connection.transports[index].move as TransportInfo).range.from].departure.time).getTime()) + ')'}</span>
                    </div>
                    <div className={isCollapsed ? 'intermediate-stops collapsed' : 'intermediate-stops expanded'}>
                        {stopGenerator(connection.stops)}
                    </div>
                    <div className="last-stop">
                        <div className="stop past">
                            <div className="timeline train-color-border"></div>
                            <div className="time">
                                <span className="past">{displayTime(connection.stops[(connection.transports[index].move as TransportInfo).range.to].arrival.time)}</span>
                            </div>
                            <div className="delay"></div>
                            <div className="station">
                                <span className="virtual">{connection.stops[(connection.transports[index].move as TransportInfo).range.to].station.name}</span>
                            </div>
                        </div>
                    </div>
                </div>)
        }
    }
    return transDivs;
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
        case 10:
            return '#ship';
            break;
        case 11:
            return '#bus';
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

export const ConnectionRender: React.FC<{ 'connection': Connection, 'setDetailViewHidden': React.Dispatch<React.SetStateAction<Boolean>> }> = (props) => {

    return (
        <svg width='335' height='40' viewBox='0 0 335 40'>
            <g>
                {isArrLengthOne(props.connection.transports) ?
                    isTransportInfo(props.connection.transports[0]) ?
                        <g className={'part train-class-' + (props.connection.transports[0].move as TransportInfo).category_id + ' acc-0'}>
                            <line x1='0' y1='12' x2='326' y2='12' className='train-line'></line>
                            <circle cx='4' cy='12' r='12' className='train-circle'></circle>
                            <use xlinkHref={classToId((props.connection.transports[0].move as TransportInfo).category_id)} className='train-icon' x='-4' y='4' width='16' height='16'></use>
                            <text x='-6' y='40' textAnchor='start' className='train-name'>{(props.connection.transports[0].move as TransportInfo).name}</text>
                            <rect x='0' y='0' width='323' height='24' className='tooltipTrigger'></rect>
                        </g>
                        :
                        <g className='part train-class-walk acc-0'>
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

export const JourneyRender: React.FC<{ 'connection': Connection, 'setSubOverlayHidden': React.Dispatch<React.SetStateAction<Boolean>>, 'setTrainSelected': React.Dispatch<React.SetStateAction<TripId>> }> = (props) => {

    const [isIntermediateStopsCollapsed, setIsIntermediateStopsCollapsed] = useState<Boolean>(true);

    const [walkTime, setWalkTime] = useState<number>(0);

    const [latStart, setLatStart] = useState<number>(50);

    const [lngStart, setLngStart] = useState<number>(10);

    const [latDest, setLatDest] = useState<number>(50);

    const [lngDest, setLngDest] = useState<number>(10);

    const [durationLimit, setDurationLimit] = useState<number>(900);

    const [profile, setProfile] = useState<string>('default');

    const [includeEdges, setIncludeEdges] = useState<boolean>(false);

    const [includePath, setIncludePath] = useState<boolean>(true);

    const [includeSteps, setIncludeSteps] = useState<boolean>(true);

    useEffect(() => {
        let requestURL = 'https://europe.motis-project.de/?elm=FootRoutingRequest';
        fetch(requestURL, getWalkTime(latStart, lngStart, latDest, lngDest, durationLimit, profile, includeEdges, includePath, includeSteps))
            .then(res => res.json())
            .then((res: FootRouting) => {
                console.log('Foot Request successful');
                console.log(res);
                setWalkTime(res.content.routes[0].routes[0].duration);
            });
    }, []);

    return (
        (props.connection === undefined) ?
            <></> :
            <>
                {isArrLengthOne(props.connection.transports) ?
                    <div className={'train-detail train-class-' + (props.connection.transports[0].move as TransportInfo).category_id}>
                        <div className='top-border'></div>
                        <div>
                            <div className={'train-box train-class-' + (props.connection.transports[0].move as TransportInfo).category_id + ' with-tooltip'}
                                data-tooltip={'Betreiber: DB Regio AG S-Bahn Rhein-Main \nZugnummer: ' + (props.connection.transports[0].move as TransportInfo).train_nr} onClick={() => { props.setSubOverlayHidden(false); props.setTrainSelected(props.connection.trips[0].id); }}>
                                <svg className='train-icon'>
                                    <use xlinkHref={classToId((props.connection.transports[0].move as TransportInfo).category_id)}></use>
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
                    transportDivs(props.connection, isIntermediateStopsCollapsed, setIsIntermediateStopsCollapsed, props.setSubOverlayHidden, props.setTrainSelected, walkTime)}
            </>
    );
};