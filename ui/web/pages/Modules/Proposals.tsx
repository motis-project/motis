import React from 'react';
import { Station } from './ConnectionTypes';

import { Address, Region } from './SuggestionTypes';


const getDisplayRegion = (regions: Region[]) => {
    console.log(regions)
    let res = regions.filter((r: Region) => r.adminLevel <= 8);
    console.log('Bin ein Filter res');
    console.log(res);
    return res[0].name + ', ' + res[-1].name;
}


export const Proposals: React.FC<{'addresses': Address[], 'stations': Station[]}> = (props) => {
    return (
        <div className='proposals'>
            {
                props.stations.map((station: Station) => (
                    <li className='' title={station.name}>
                        <i className='icon'>train</i>
                        <span className='station'>{station.name}</span>
                    </li>
                ))}
            {
                props.addresses.map((address: Address) => (
                    <li className='' title='SFS Saar, Bexbach'>
                        <i className='icon'>place</i>
                        <span className='address-name'>{address.name}</span>
                        <span className='address-region'>{address.name}</span>
                    </li>
                ))}
        </div>
    )
}