import moment from 'moment';
import React, { useState } from 'react';

import { TransportInfo, Connection } from '../Types/Connection';


export const classToId = (classz: Number) => {
    switch (classz) {
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
        default:
            return '#bus';
    }
}

let toolTipID = 0;
let xPos = [];

const transportForLoop = (connection: Connection, setToolTipSelected: React.Dispatch<React.SetStateAction<number>>) => {
    let elements = [];
    let percentage = 0;
    let durationWalk = (connection.transports.length > 1 && connection.transports[1].move_type === 'Walk') ?
        moment.unix(connection.stops[connection.transports[1].move.range.to].arrival.time).diff(moment.unix(connection.stops[connection.transports[1].move.range.from].departure.time)) :
        0;
    let durationTransportPartial = 0;
    let durationTransportFull = moment.unix(connection.stops[connection.stops.length - 1].arrival.time).diff(moment.unix(connection.stops[0].departure.time));
    let prevLength = 0;
    for (let index = 0; index < connection.transports.length; index++) {
        if (connection.transports[index].move_type === 'Transport') {
            durationTransportPartial = moment.unix(connection.stops[connection.transports[index].move.range.to].arrival.time)
                .diff(moment.unix(connection.stops[connection.transports[index].move.range.from].departure.time));
            percentage = (durationTransportPartial + durationWalk) / durationTransportFull;
            elements.push(
                <g className={'part train-class-' + (connection.transports[index].move as TransportInfo).clasz + ' acc-0'} key={toolTipID}>
                    <line x1={prevLength} y1='12' x2={((percentage * 326 + prevLength) > 326) ? 326 : (percentage * 326 + prevLength)} y2='12' className='train-line'></line>
                    <circle cx={prevLength + 4} cy='12' r='12' className='train-circle'></circle>
                    <use xlinkHref={classToId((connection.transports[index].move as TransportInfo).clasz)} className='train-icon' x={prevLength - 4} y='4' width='16' height='16'></use>
                    <text x={prevLength - 6} y='40' textAnchor='start' className='train-name'>{(connection.transports[index].move as TransportInfo).name}</text>
                    <rect x={prevLength} y='0' width={(percentage * 326 + prevLength)} height='24' className='tooltipTrigger' onMouseOver={() => { setToolTipSelected(index)}} onMouseOut={() => { setToolTipSelected(-1)}}></rect>
                </g>
            );
            xPos.push(prevLength);
            prevLength = prevLength + (percentage * 326);
            toolTipID += 1;
        } else {
            if (index > 1) {
                durationWalk = moment.unix(connection.stops[connection.transports[index].move.range.to].arrival.time).diff(moment.unix(connection.stops[connection.transports[index].move.range.from].departure.time))
            }
        }
    }
    toolTipID = 0;
    return elements;
}


const toolTipGenerator = (connection: Connection, toolTipSelected: number) => {
    let toolTips = [];
    let counter = 0;
    for (let index = 0; index < connection.transports.length; index++) {
        if (connection.transports[index].move_type === 'Transport') {
            let offset = ((Number(xPos[counter]) + 240) > 335) ? 95 : xPos[counter];
            counter += 1;
            toolTips.push(
                <div className={(index === toolTipSelected) ? 'tooltip visible' : 'tooltip'} style={{ position: 'absolute', left: offset + 'px', top: '23px' }} key={index}>
                    <div className='stations'>
                        <div className='departure'>
                            <span className='station'>{connection.stops[connection.transports[index].move.range.from].station.name}</span>
                            <span className='time'>{moment.unix(connection.stops[connection.transports[index].move.range.from].departure.time).format('HH:mm')}</span>
                        </div>
                        <div className='arrival'>
                            <span className='station'>{connection.stops[connection.transports[index].move.range.to].station.name}</span>
                            <span className='time'>{moment.unix(connection.stops[connection.transports[index].move.range.to].arrival.time).format('HH:mm')}</span>
                        </div>
                    </div>
                    <div className='transport-name'>
                        <span>{(connection.transports[index].move as TransportInfo).name}</span>
                    </div>
                </div>
            );
            toolTipID += 1
        }
    }
    toolTipID = 0;
    xPos = [];
    return toolTips;
}


export const ConnectionRender: React.FC<{ 'connection': Connection, 'setDetailViewHidden': React.Dispatch<React.SetStateAction<Boolean>> }> = (props) => {

    const [toolTipSelected, setToolTipSelected] = useState<number>(-1)

    return (
        <>
            <svg width='335' height='40' viewBox='0 0 335 40'>
                <g>
                    {(props.connection.transports[0].move_type === 'Transport') ?
                        transportForLoop(props.connection, setToolTipSelected)
                        :
                        <g className='part train-class-walk acc-0'>
                            <line x1='0' y1='12' x2='326' y2='12' className='train-line'></line>
                            <circle cx='4' cy='12' r='12' className='train-circle'></circle>
                            <use xlinkHref='#walk' className='train-icon' x='-4' y='4' width='16' height='16'></use>
                            <text x='0' y='40' textAnchor='start' className='train-name'></text>
                            <rect x='0' y='0' width='323' height='24' className='tooltipTrigger'></rect>
                        </g>
                    }
                </g>
                <g className='destination'><circle cx='329' cy='12' r='6'></circle></g>
            </svg>
            {toolTipGenerator(props.connection, toolTipSelected)}
        </>
    );
};