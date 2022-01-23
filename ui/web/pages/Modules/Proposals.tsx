import React from 'react';

import { Station } from './ConnectionTypes';
import { Address, Region } from './SuggestionTypes';


const getDisplayRegions = (regions: Region[]) => {
    let res = regions.filter((r: Region) => r.admin_level <= 8);
    return res[0].name + ', ' + res[res.length - 1].name;
}


export const StationSuggestion: React.FC<{'station': Station, 'setName': React.Dispatch<React.SetStateAction<string>>, 'setSuggestion': React.Dispatch<React.SetStateAction<Station | Address>>, 'setShowSuggestions': React.Dispatch<React.SetStateAction<boolean>>}> = (props) => {

    return (
        <li className='' 
            title={props.station.name} 
            onClick={() => {props.setName(props.station.name); 
                            props.setSuggestion(props.station); 
                            props.setShowSuggestions(false)}}
            >
            <i className='icon'>train</i>
            <span className='station'>{props.station.name}</span>
        </li>
    )
}


export const AddressSuggestion: React.FC<{'address': Address, 'setName': React.Dispatch<React.SetStateAction<string>>, 'setSuggestion': React.Dispatch<React.SetStateAction<Station | Address>>, 'setShowSuggestions': React.Dispatch<React.SetStateAction<boolean>>}> = (props) => {

    return (
        <li className='' 
            title={props.address.name + ', ' + getDisplayRegions(props.address.regions).split(',')[0]} 
            onClick={() => {props.setName(props.address.name); 
                            props.setSuggestion(props.address);
                            props.setShowSuggestions(false) }}
            >
            <i className='icon'>place</i>
            <span className='address-name'>{props.address.name}</span>
            <span className='address-region'>{getDisplayRegions(props.address.regions)}</span>
        </li>
    )
}