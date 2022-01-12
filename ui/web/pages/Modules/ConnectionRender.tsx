import React from 'react';
import Index from '..';
import { TransportInfo, WalkInfo } from './ConnectionTypes';

const isTransportInfo = (transport: TransportInfo | WalkInfo): transport is TransportInfo => {
    return (transport as TransportInfo).name !== undefined;
}

export const ConnectionRender: React.FC<{ 'transports': (TransportInfo | WalkInfo)[] }> = (props) => {
    return (
        <svg width='335' height='40' viewBox='0 0 335 40'>
            {props.transports.map((transport: TransportInfo | WalkInfo, index) => (
                isTransportInfo(transport) ?
                    <g className={'part train-class-' + transport.category_id + ' acc-0'} key={index}>
                        <line x1='0' y1='12' x2='326' y2='12' className='train-line'></line>
                        <circle cx='12' cy='12' r='12' className='train-circle'></circle>
                        <use xlinkHref='#train' className='train-icon' x='4' y='4' width='16' height='16'></use>
                        <text x='0' y='40' textAnchor='start' className='train-name'></text>
                        <rect x='0' y='0' width='323' height='24' className='tooltipTrigger'></rect>
                    </g>
                    :
                    <g className={'part train-class-walk acc-0'} key={index}>
                        <line x1='0' y1='12' x2='326' y2='12' className='train-line'></line>
                        <circle cx='12' cy='12' r='12' className='train-circle'></circle>
                        <use xlinkHref='#train' className='train-icon' x='4' y='4' width='16' height='16'></use>
                        <text x='0' y='40' textAnchor='start' className='train-name'></text>
                        <rect x='0' y='0' width='323' height='24' className='tooltipTrigger'></rect>
                    </g>
            ))}
            <g className="destination"><circle cx="329" cy="12" r="6"></circle></g>
        </svg>
    );
};