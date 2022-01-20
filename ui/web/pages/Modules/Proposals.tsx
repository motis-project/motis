import React from 'react';
import { Station } from './ConnectionTypes';

import { Address, Region } from './SuggestionTypes';


const getDisplayRegion = (regions: Region[]) => {
    let res = regions.filter((r: Region) => r.admin_level <= 8);
    return res[0].name + ', ' + res[res.length - 1].name;
}


export const Proposals: React.FC<{'addresses': Address[], 'stations': Station[]}> = (props) => {
    return (
        <div className='proposals'>
            {
                props.stations.map((station: Station, index: number) => (
                    <li className='' title={station.name} key={index}>
                        <i className='icon'>train</i>
                        <span className='station'>{station.name}</span>
                    </li>
                ))}
            {
                props.addresses.map((address: Address, index: number) => (
                    <li className='' title='SFS Saar, Bexbach' key={index}>
                        <i className='icon'>place</i>
                        <span className='address-name'>{address.name}</span>
                        <span className='address-region'>{getDisplayRegion(address.regions)}</span>
                    </li>
                ))}
        </div>
    )
}