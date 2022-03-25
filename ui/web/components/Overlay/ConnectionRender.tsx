import moment from 'moment';
import React, { useEffect, useState } from 'react';

import { TransportInfo, Connection, Transport, WalkInfo, Stop } from '../Types/Connection';


export const classToId = (transport: Transport) => {
    switch (getClasz(transport)) {
        case 0:
            return '#plane';
        case 1:
            return '#train';
        case 2:
            return '#train';
        case 3:
            return '#bus';
        case 4:
            return '#train';
        case 5:
            return '#train';
        case 6:
            return '#train';
        case 7:
            return '#sbahn';
        case 8:
            return '#ubahn';
        case 9:
            return '#tram';
        case 11:
            return '#ship';
        case 'walk':
            return '#walk';
        case 'bike':
            return '#bike';
        case 'car':
            return '#car';
        default:
            return '#bus';
    }
}

const getAccNumber = (transport: Transport) => {
    if (transport.move_type === 'Walk') {
        let walkInfo = transport.move as WalkInfo;
        if (walkInfo.accessibility === 0) {
            return 'acc-0';
        } else if (walkInfo.accessibility < 30) {
            return 'acc-1';
        } else {
            return 'acc-2';
        }
    } else {
        return 'acc-0';
    }
}

const getClasz = (transport: Transport) => {
    switch (transport.move_type) {
        case 'Transport':
            return (transport.move as TransportInfo).clasz;
            break;
        case 'Walk':
            if((transport.move as WalkInfo).mumo_type === ''){
                return 'walk'
            }
            return (transport.move as WalkInfo).mumo_type;
            break;
        default:
            return '';
            break;
    }
}

const calcPartWidth = (transport: Transport, totalDurationInMill: number, stops: Stop[], totalWidth: number) => {
    let baseBarLength = 2;
    let avgCharLength = 7;
    let trainNameLength = (transport.move_type === 'Transport') ? ((transport.move as TransportInfo).name.length * avgCharLength) : 0;
    let transportTimeInMill = moment.unix(stops[transport.move.range.to].arrival.time).diff(moment.unix(stops[transport.move.range.from].departure.time));
    let percentage = transportTimeInMill / totalDurationInMill;
    let partWidth = (percentage >= 1) ? totalWidth : ((percentage * totalWidth) + baseBarLength);
    let minWidth = 26;
    return Math.max(Math.max(trainNameLength, partWidth), minWidth);
}

const getTransportTime = (transport: Transport, stops: Stop[]) => {
    if (transport.move_type === 'Transport') {
        return moment.unix(stops[transport.move.range.to].arrival.time).diff(moment.unix(stops[transport.move.range.from].departure.time));
    }

    return 0;
}

interface PartElem {
    transport: Transport,
    position: number,
    partWidth: number,
    lineEnd: number,
    classId: string,
    trainName?: string,
    clasz: (string | number),
    acc: string
}

export const ConnectionRender: React.FC<{ 'connection': Connection, 'setDetailViewHidden': React.Dispatch<React.SetStateAction<Boolean>>, 'setConnectionHighlighted': React.Dispatch<React.SetStateAction<boolean>>, 'connectionDoNothing': boolean, 'connectionHighlighted': boolean }> = (props) => {

    const [toolTipSelected, setToolTipSelected] = useState<string>('');
    const [parts, setParts] = useState<PartElem[]>([]);

    useEffect(() => {
        let p: PartElem[] = [];
        let position = 0;

        props.connection.transports.map((transport: Transport, index) => {
            let partWidth = calcPartWidth(transport, totalDurationInMill, props.connection.stops, totalWidth - (destinationRadius * 2));
            let lineEnd = position + partWidth + (destinationRadius / 2);
            if(lineEnd > 323){
                lineEnd = 323;
            }
            let classId = classToId(transport);
            let clasz = getClasz(transport);
            let acc = getAccNumber(transport);
            console.log(`transport ${transport}`)

            if (transport.move_type === 'Transport') {
                let trainName = (transport.move as TransportInfo).name;
                p.push({ transport: transport, position: position, partWidth: partWidth, lineEnd: lineEnd, classId: classId, trainName: trainName, clasz: clasz, acc: acc });
                position += partWidth;
            } else if ((index === 0 || index === props.connection.transports.length - 1) && (transport.move_type === 'Walk')) {
                p.push({ transport: transport, position: position, partWidth: partWidth, lineEnd: lineEnd, classId: classId, clasz: clasz, acc: acc });
                position += partWidth;
            }
        })
        setParts(p);
    }, [])

    let initialValue = 0;
    let totalDurationInMill = props.connection.transports.reduce((previousValue, currentValue) => previousValue + getTransportTime(currentValue, props.connection.stops), initialValue);
    //Variablen die im originalen auch verwendet wurden, teilweise weggelassen
    let iconSize = 16;
    let circleRadius = 12;
    let basePartSize = circleRadius * 2;
    let iconOffset = ((circleRadius * 2) - iconSize) / 2;
    let destinationRadius = 6;
    let textOffset = circleRadius * 2 + 4;
    let textHeight = 12;
    let totalHeight = textOffset + textHeight;
    let totalWidth = 335; //transportListViewWidth aus Connections.elm
    let tooltipWidth = 240;

    return (
        <>
            <svg width={totalWidth} height={totalHeight} viewBox={`0 0 ${totalWidth} ${totalHeight}`}>
                <g>
                    {parts.map((partElem: PartElem, index) => (
                        <g className={`part train-class-${partElem.clasz} ${partElem.acc}${(props.connectionDoNothing) ? '' : (props.connectionHighlighted) ? 'highlighted' : 'faded'}`} key={index}> {/*Die Abfrage nach dem highlight muss anders sein iwas mit line id*/}
                            <line x1={partElem.position} y1={circleRadius} x2={partElem.lineEnd} y2={circleRadius} className='train-line'></line>
                            <circle cx={partElem.position + circleRadius} cy={circleRadius} r={circleRadius} className='train-circle' ></circle>
                            <use xlinkHref={partElem.classId} className='train-icon' x={partElem.position + iconOffset} y={iconOffset} width={iconSize} height={iconSize} ></use>
                            <text x={partElem.position} y={textOffset + textHeight} textAnchor='start' className='train-name'>{partElem.trainName}</text>
                            <rect x={partElem.position} y='0' width={partElem.position + partElem.partWidth} height={basePartSize} className='tooltipTrigger'
                                onMouseOver={() => { setToolTipSelected((partElem.transport.move as TransportInfo).line_id) }}
                                onMouseOut={() => { setToolTipSelected('') }}></rect>
                        </g>
                    ))}
                </g>
                <g className='destination'><circle cx={totalWidth - destinationRadius} cy={circleRadius} r={destinationRadius}></circle></g>
            </svg>
            {parts.map((partElem: PartElem, index) => (
                <div className={`tooltip ${(toolTipSelected === (partElem.transport.move as TransportInfo).line_id) ? 'visible' : ''}`} style={{ position: 'absolute', left: `${(Math.min(partElem.position, (totalWidth - tooltipWidth)))}px`, top: `${(textOffset - 5)}px` }} key={(partElem.transport.move as TransportInfo).line_id}>
                    <div className='stations'>
                        <div className='departure'>
                            <div className='station'>
                                {props.connection.stops[partElem.transport.move.range.from].station.name}
                            </div>
                            <div className='time'>
                                {moment.unix(props.connection.stops[partElem.transport.move.range.from].departure.time).format('HH:mm')}
                            </div>
                        </div>
                        <div className='arrival'>
                            <div className='station'>
                                {props.connection.stops[partElem.transport.move.range.to].station.name}
                            </div>
                            <div className='time'>
                                {moment.unix(props.connection.stops[partElem.transport.move.range.to].arrival.time).format('HH:mm')}
                            </div>
                        </div>
                    </div>
                    <div className='transport-name'>
                        {partElem.trainName}
                    </div>
                </div>
            ))}
        </>
    );
};