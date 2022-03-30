import moment from 'moment';
import React, { useEffect, useState } from 'react';
import { Translations } from '../App/Localization';

import { TransportInfo, Connection, Transport, WalkInfo, Stop, Trip, TripId } from '../Types/Connection';


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
        case 'walk' || 'foot':
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

export const getClasz = (transport: Transport) => {
    switch (transport.move_type) {
        case 'Transport':
            return (transport.move as TransportInfo).clasz;
            break;
        case 'Walk':
            if ((transport.move as WalkInfo).mumo_type === '' || (transport.move as WalkInfo).mumo_type === 'foot') {
                return 'walk'
            }
            return (transport.move as WalkInfo).mumo_type;
            break;
        default:
            return '';
            break;
    }
}

const calcPartWidths = (transports: Transport[], totalDurationInMill: number, stops: Stop[], totalWidth: number, destinationRadius: number) => {
    let partWidths: GraphData[] = [];
    //constants from motis-project
    let baseBarLength = 2;
    let avgCharLength = 7;
    let minWidth = 26;

    let percentage = 0;
    let finalWidth = 0;
    let requiredWidth = 0;
    let availableWidth = totalWidth;
    let position = 0;
    let lineEnd = 0;
    let final = false;

    transports.map((t: Transport, index) => {
        if ((t.move_type === 'Transport') || ((index === 0 || index === transports.length - 1) && (t.move_type === 'Walk'))) {
            let trainNameLength = (t.move_type === 'Transport') ? ((t.move as TransportInfo).name.length * avgCharLength) : 0;

            let transportTimeInMill = moment.unix(stops[t.move.range.to].arrival.time).diff(moment.unix(stops[t.move.range.from].departure.time));
            percentage = transportTimeInMill / totalDurationInMill;

            let partWidth = (percentage >= 1) ? totalWidth : ((percentage * totalWidth) + baseBarLength);

            finalWidth = Math.max(Math.max(trainNameLength, partWidth), minWidth);
            requiredWidth += finalWidth;
            if (finalWidth == minWidth) {
                final = true;
                availableWidth -= minWidth;
            }
            lineEnd = position + finalWidth + (destinationRadius / 2);

            partWidths.push({ position: position, partWidth: finalWidth, lineEnd: lineEnd, percentage: percentage, final: final });
            position += finalWidth;
            final = false;
        }
    });

    if (totalWidth < requiredWidth) {
        console.log(transports);
        console.log(partWidths);
        return calcFinalPartWidths(partWidths, availableWidth);
    } else if (totalWidth > requiredWidth) {
        let remainingWidth = totalWidth - requiredWidth;
        let tmp: GraphData[] = [];
        let newPosition = 0;
        partWidths.map((g: GraphData, index) => {
            let newWidth = g.partWidth + Math.ceil(g.percentage * remainingWidth); //adds the remaining width proportionally to the existing partWidth
            let newLineEnd = (newPosition + newWidth + (destinationRadius / 2));
            if (index == partWidths.length - 1 && newLineEnd < 323) {
                newLineEnd = 323;
            }
            tmp.push({ position: newPosition, partWidth: newWidth, lineEnd: newLineEnd, percentage: g.percentage, final: true });
            newPosition += newWidth;
        })
        return tmp;
    } else if (totalWidth == requiredWidth) {
        return partWidths;
    }
    return [];
}

const calcFinalPartWidths = (partWidths: GraphData[], totalWidth: number) => {
    let newPartWidths: GraphData[] = [];
    let partWidthCopy = [...partWidths];
    let availableWidth = totalWidth;
    let requiredWidth = 0;
    let newPosition = 0;
    let allFinal = false;

    while (allFinal === false) {
        allFinal = true;
        newPartWidths = [];
        newPosition = 0;
        partWidthCopy.map((g: GraphData) => {
            if (g.final) {
                newPartWidths.push({ position: newPosition, partWidth: g.partWidth, lineEnd: (newPosition + g.partWidth + 3), percentage: g.percentage, final: true });
                newPosition += g.partWidth;
                requiredWidth += g.partWidth;
            } else {
                let newWidth = g.percentage * availableWidth;
                let final = false;
                if (newWidth < 26) {
                    newWidth = 26;
                    availableWidth -= 26;
                    final = true;
                    allFinal = false;
                }
                newPartWidths.push({ position: newPosition, partWidth: newWidth, lineEnd: (newPosition + newWidth + 3), percentage: g.percentage, final: final });
                newPosition += newWidth;
                requiredWidth += newWidth;
            }
        });
        if(requiredWidth <= 323){
            break;
        }
        console.log(newPartWidths);
        partWidthCopy = newPartWidths;
    }
    if (requiredWidth < 323) {
        let remainingWidth = 323 - requiredWidth;
        let tmp: GraphData[] = [];
        let newPosition = 0;
        newPartWidths.map((g: GraphData, index) => {
            let newWidth = g.partWidth + Math.ceil(g.percentage * remainingWidth); //adds the remaining width proportionally to the existing partWidth
            let newLineEnd = (newPosition + newWidth + 3);
            if (index == partWidths.length - 1 && newLineEnd !== 323) {
                newLineEnd = 323;
            }
            tmp.push({ position: newPosition, partWidth: newWidth, lineEnd: newLineEnd, percentage: g.percentage, final: true });
            newPosition += newWidth;
        });
        newPartWidths = tmp;
    }
    return newPartWidths;
}

interface PartElem {
    transport: Transport,
    graphData: GraphData,
    classId: string,
    trainName?: string,
    transportName?: string,
    clasz: (string | number),
    acc: string,
    trainNumber?: number
}

interface GraphData {
    position: number,
    partWidth: number,
    lineEnd: number,
    percentage: number,
    final: boolean
}

export const ConnectionRender: React.FC<{ 'translation': Translations, 'connection': Connection, 'connectionHighlighted': boolean, 'mapData': any, 'parentIndex': number }> = (props) => {

    const [toolTipSelected, setToolTipSelected] = useState<number>(-1);
    const [parts, setParts] = useState<PartElem[]>([]);
    const [partsHighlighted, setPartsHighlighted] = useState<number[]>([]);

    let totalDurationInMill = moment.unix(props.connection.stops[props.connection.stops.length - 1].arrival.time).diff(moment.unix(props.connection.stops[0].departure.time));
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

    useEffect(() => {
        let p: PartElem[] = [];
        let g: GraphData[] = calcPartWidths(props.connection.transports, totalDurationInMill, props.connection.stops, totalWidth - (destinationRadius * 2), destinationRadius);
        let counter = 0;

        props.connection.transports.map((transport: Transport, index) => {
            let classId = classToId(transport);
            let clasz = getClasz(transport);
            let acc = getAccNumber(transport);
            let trainName = '';
            let transportName = '';
            if (transport.move_type === 'Transport') {
                trainName = (transport.move as TransportInfo).name;
            } else {
                switch ((transport.move as WalkInfo).mumo_type) {
                    case 'car':
                        transportName = props.translation.connections.car.toString();
                        break;
                    case 'bike':
                        transportName = props.translation.connections.bike.toString();
                        break;
                    default:
                        transportName = props.translation.connections.walk.toString();
                        break;
                }
            }

            if (transport.move_type === 'Transport') {
                p.push({ transport: transport, graphData: g[counter], classId: classId, trainName: trainName, clasz: clasz, acc: acc, trainNumber: (transport.move as TransportInfo).train_nr });
                counter += 1;
            } else if ((index === 0 || index === props.connection.transports.length - 1) && (transport.move_type === 'Walk')) {
                p.push({ transport: transport, graphData: g[counter], classId: classId, transportName: transportName, clasz: clasz, acc: acc });
                counter += 1;
            }
        })
        setParts(p);
    }, [])

    useEffect(() => {
        let tmp = [];
        if (props.mapData !== undefined && props.mapData.hoveredTripSegments !== null) {
            props.mapData.hoveredTripSegments.map((elem: any) => {
                tmp.push(elem.trip[0].train_nr); //walkinfos werden nicht beachtet!
            });
        }
        setPartsHighlighted(tmp);
    }, [props.mapData])


    return (
        <>
            <svg width={totalWidth} height={totalHeight} viewBox={`0 0 ${totalWidth} ${totalHeight}`}>
                <g>
                    {parts.map((partElem: PartElem) => (
                        <g className={`part train-class-${partElem.clasz} ${partElem.acc} ${(props.connectionHighlighted) ? ((partsHighlighted.includes(partElem.trainNumber)) ? 'highlighted' : 'faded') : ''}`} key={`${props.parentIndex}_${partElem.trainNumber}`}>
                            <line x1={partElem.graphData.position} y1={circleRadius} x2={partElem.graphData.lineEnd} y2={circleRadius} className='train-line'></line>
                            <circle cx={partElem.graphData.position + circleRadius} cy={circleRadius} r={circleRadius} className='train-circle' ></circle>
                            <use xlinkHref={partElem.classId} className='train-icon' x={partElem.graphData.position + iconOffset} y={iconOffset} width={iconSize} height={iconSize} ></use>
                            <text x={partElem.graphData.position} y={textOffset + textHeight} textAnchor='start' className='train-name'>{partElem.trainName}</text>
                            <rect x={partElem.graphData.position} y='0' width={partElem.graphData.position + partElem.graphData.partWidth} height={basePartSize} className='tooltipTrigger'
                                onMouseOver={() => { setToolTipSelected(partElem.trainNumber) }}
                                onMouseOut={() => { setToolTipSelected(-1) }}></rect>
                        </g>
                    ))}
                </g>
                <g className='destination'><circle cx={totalWidth - destinationRadius} cy={circleRadius} r={destinationRadius}></circle></g>
            </svg>
            {parts.map((partElem: PartElem, index) => (

                <div className={`tooltip ${((toolTipSelected === partElem.trainNumber) || (partsHighlighted.includes(partElem.trainNumber))) ? 'visible' : ''}`} style={{ position: 'absolute', left: `${(Math.min(partElem.graphData.position, (totalWidth - tooltipWidth)))}px`, top: `${(textOffset - 5)}px` }} key={`tooltip${props.parentIndex}${index}`}>

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
                        {(partElem.transport.move_type === 'Transport') ? partElem.trainName : partElem.transportName}
                    </div>
                </div>
            ))}
        </>
    );
};