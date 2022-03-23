import React from 'react';

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


export const ConnectionRender: React.FC<{ 'connection': Connection, 'setDetailViewHidden': React.Dispatch<React.SetStateAction<Boolean>> }> = (props) => {

    return (
        <svg width='335' height='40' viewBox='0 0 335 40'>
            <g>
                {props.connection.transports.length === 1 ?
                    isTransportInfo(props.connection.transports[0]) ?
                        <g className={'part train-class-' + (props.connection.transports[0].move as TransportInfo).clasz + ' acc-0'}>
                            <line x1='0' y1='12' x2='326' y2='12' className='train-line'></line>
                            <circle cx='4' cy='12' r='12' className='train-circle'></circle>
                            <use xlinkHref={classToId((props.connection.transports[0].move as TransportInfo).clasz)} className='train-icon' x='-4' y='4' width='16' height='16'></use>
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