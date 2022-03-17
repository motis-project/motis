import moment from 'moment';
import React, { useState } from 'react';

import { Transport, TransportInfo, Connection } from '../Types/Connection';


const isTransportInfo = (transport: Transport) => {
    return transport.move_type === 'Transport';
}


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

const transportForLoop = (connection: Transport[], setToolTipSelected: React.Dispatch<React.SetStateAction<number>>) => {
    let elements = [];
    let percentage = 0;
    let rangeMax = connection[connection.length - 1].move.range.to;
    let walk = 0;
    let prevLength = 0;
    for (let index = 0; index < connection.length; index++) {
        percentage = (connection[index].move.range.to - connection[index].move.range.from + walk) / rangeMax;
        isTransportInfo(connection[index]) ?
            elements.push(
                <g className={'part train-class-' + (connection[index].move as TransportInfo).clasz + ' acc-0'} key={toolTipID}>
                    <line x1={prevLength} y1='12' x2={(percentage * 326 + prevLength)} y2='12' className='train-line'></line>
                    <circle cx={prevLength + 4} cy='12' r='12' className='train-circle'></circle>
                    <use xlinkHref={classToId((connection[index].move as TransportInfo).clasz)} className='train-icon' x={prevLength - 4} y='4' width='16' height='16'></use>
                    <text x={prevLength - 6} y='40' textAnchor='start' className='train-name'>{(connection[index].move as TransportInfo).name}</text>
                    <rect x={prevLength} y='0' width={(percentage * 326 + prevLength)} height='24' className='tooltipTrigger' onMouseOver={() => { setToolTipSelected(toolTipID); console.log(toolTipID) }} onMouseOut={() => { setToolTipSelected(-1); console.log(toolTipID) }}></rect>
                </g>
            ) :
            walk = 1;
        console.log('transportForLoop');
        console.log(toolTipID);
        toolTipID += 1;
        if (isTransportInfo(connection[index])) {
            prevLength = prevLength + (percentage * 326);
        }
    }
    toolTipID = 0;
    return elements;
}


const toolTipGenerator = (connection: Connection, toolTipSelected: number) => {
    let toolTips = [];
    let offset = 0;

    for (let index = 0; index < connection.transports.length; index++) {
        if (isTransportInfo(connection.transports[index])) {
            toolTips.push(
                <div className={(toolTipID === toolTipSelected) ? 'tooltip visible' : 'tooltip'} style={{ position: 'absolute', left: offset + 'px', top: '23px' }} key={toolTipID}>
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
            offset += 50
            console.log('toolTipGenerator');
            console.log(toolTipID);
            toolTipID += 1
        }
    }
    toolTipID = 0;
    return toolTips;
}


export const ConnectionRender: React.FC<{ 'connection': Connection, 'setDetailViewHidden': React.Dispatch<React.SetStateAction<Boolean>> }> = (props) => {

    const [toolTipSelected, setToolTipSelected] = useState<number>(-1)

    return (
        <>
            <svg width='335' height='40' viewBox='0 0 335 40'>
                <g>
                    {isTransportInfo(props.connection.transports[0]) ?
                        transportForLoop(props.connection.transports, setToolTipSelected)
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