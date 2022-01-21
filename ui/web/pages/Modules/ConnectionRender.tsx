import React from 'react';
import Index from '..';
import { Transport, TransportInfo, WalkInfo } from './ConnectionTypes';

const isTransportInfo = (transport: Transport) => {
    console.log(transport)
    console.log((transport.move as TransportInfo).category_id)
    return transport.move_type === 'Transport';
}

export const ConnectionRender: React.FC<{ 'transports': Transport[] }> = (props) => {
    return (
        <svg width='335' height='40' viewBox='0 0 335 40'>
            <g>
                {props.transports.map((transport: Transport, index) => (
                    isTransportInfo(transport) ?
                        <g className={'part train-class-' + (transport.move as TransportInfo).category_id + ' acc-0'} key={index}>
                            <line x1='0' y1='12' x2='326' y2='12' className='train-line'></line>
                            <circle cx='12' cy='12' r='12' className='train-circle'></circle>
                            <use xlinkHref='#train' className='train-icon' x='4' y='4' width='16' height='16'></use>
                            <text x='0' y='40' textAnchor='start' className='train-name'></text>
                            <rect x='0' y='0' width='323' height='24' className='tooltipTrigger'></rect>
                        </g>
                        :
                        <g className='part train-class-walk acc-0' key={index}>
                            <line x1='0' y1='12' x2='326' y2='12' className='train-line'></line>
                            <circle cx='12' cy='12' r='12' className='train-circle'></circle>
                            <use xlinkHref='#walk' className='train-icon' x='4' y='4' width='16' height='16'></use>
                            <text x='0' y='40' textAnchor='start' className='train-name'></text>
                            <rect x='0' y='0' width='323' height='24' className='tooltipTrigger'></rect>
                        </g>
                ))}
            </g>
            <g className='destination'><circle cx='329' cy='12' r='6'></circle></g>
        </svg>
    );
};