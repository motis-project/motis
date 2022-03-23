import React from 'react';

import { Station } from '../Types/Connection';
import { setLocalStorage } from '../App/LocalStorage';
import { Address, Region } from '../Types/SuggestionTypes';


interface Proposals {
    addresses: Address[], 
    stations: Station[], 
    suggestions: (Address | Station) [], 
    showSuggestions: boolean, 
    highlighted: number, 
    localStorageStation: string, 
    inputFieldRef: React.MutableRefObject<any>
    setName: React.Dispatch<React.SetStateAction<string>>, 
    setSuggestion: React.Dispatch<React.SetStateAction<Station | Address>>, 
    setShowSuggestions: React.Dispatch<React.SetStateAction<boolean>>, 
    setSelectedSuggestion: React.Dispatch<React.SetStateAction<number>>,
    setSelected: React.Dispatch<React.SetStateAction<string>>, 
}


interface AddressSuggestions {
    address: Address, 
    listID: number, 
    selectedSuggestion: number, 
    localStorageStation: string
    setName: React.Dispatch<React.SetStateAction<string>>, 
    setSuggestion: React.Dispatch<React.SetStateAction<Station | Address>>, 
    setShowSuggestions: React.Dispatch<React.SetStateAction<boolean>>, 
    setSelectedSuggestion: React.Dispatch<React.SetStateAction<number>>, 
}


interface StationSuggestions {
    station: Station, 
    listID: number, 
    selectedSuggestion: number, 
    localStorageStation: string
    setName: React.Dispatch<React.SetStateAction<string>>, 
    setSuggestion: React.Dispatch<React.SetStateAction<Station | Address>>, 
    setShowSuggestions: React.Dispatch<React.SetStateAction<boolean>>, 
    setSelectedSuggestion: React.Dispatch<React.SetStateAction<number>>, 
}


const getDisplayRegions = (regions: Region[]) => {
    let res = regions.filter((r: Region) => r.admin_level <= 8 && r.admin_level >= 2);
    return res[0].name + ', ' + res[res.length - 1].name;
}


function useOutsideAlerter(ref: React.MutableRefObject<any>, inputFieldRef: React.MutableRefObject<any>, setShowSuggestions: React.Dispatch<React.SetStateAction<boolean>>, setSelected : React.Dispatch<React.SetStateAction<string>>) {
    React.useEffect(() => {
        /**
         * Alert if clicked on outside of element
         */
        function handleClickOutside(event) {
            if (ref.current && !ref.current.contains(event.target) && 
                inputFieldRef.current && !inputFieldRef.current.contains(event.target)) {
                setShowSuggestions(false);
                setSelected('');
            }
        }

        // Bind the event listener
        document.addEventListener("mousedown", handleClickOutside);
        return () => {
            // Unbind the event listener on clean up
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, [ref]);
}


const StationSuggestion: React.FC<StationSuggestions> = (props) => {

    return (
        <li className={props.listID == props.selectedSuggestion ? 'selected' : ''} 
            title={props.station.name} 
            onMouseDown={(e) => {
                if(e.button == 0){
                    props.setName(props.station.name); 
                    props.setSuggestion(props.station); 
                    props.setShowSuggestions(false);
                    props.setSelectedSuggestion(0);
                    setLocalStorage(props.localStorageStation, props.station);
                }
            }}
            onMouseOver={() => {
                props.setSelectedSuggestion(props.listID)
            }}
            >
            <i className='icon'>train</i>
            <span className='station'>{props.station.name}</span>
        </li>
    )
}


const AddressSuggestion: React.FC<AddressSuggestions> = (props) => {

    return (
        <li className={props.listID == props.selectedSuggestion ? 'selected' : ''}
            title={props.address.name + ', ' + getDisplayRegions(props.address.regions).split(',')[0]} 
            onMouseDown={() => {
                props.setName(props.address.name); 
                props.setSuggestion(props.address);
                props.setShowSuggestions(false);
                props.setSelectedSuggestion(0);
                setLocalStorage(props.localStorageStation, props.address);
            }}
            onMouseOver={() => {
                props.setSelectedSuggestion(props.listID)
            }}
            >
            <i className='icon'>place</i>
            <span className='address-name'>{props.address.name}</span>
            <span className='address-region'>{getDisplayRegions(props.address.regions)}</span>
        </li>
    )
}


export const Proposals: React.FC<Proposals> = (props) => {

    const suggestionsRef = React.useRef(null);

    useOutsideAlerter(suggestionsRef, props.inputFieldRef, props.setShowSuggestions, props.setSelected);

    return (
        <div className="proposals" ref={suggestionsRef}>
            {
                props.suggestions.slice(0,6).map((station: Station, index: number) => (
                    <StationSuggestion  station={station} 
                                        key={index} 
                                        listID={index}
                                        selectedSuggestion={props.highlighted}
                                        setName={props.setName} 
                                        setSuggestion={props.setSuggestion} 
                                        setShowSuggestions={props.setShowSuggestions}
                                        setSelectedSuggestion={props.setSelectedSuggestion}
                                        localStorageStation={props.localStorageStation}/>
                ))
            }
            {
                props.suggestions.slice(6).map((address: Address, index: number) => (
                    <AddressSuggestion  address={address} 
                                        key={index} 
                                        listID={index+6}
                                        selectedSuggestion={props.highlighted}
                                        setName={props.setName} 
                                        setSuggestion={props.setSuggestion} 
                                        setShowSuggestions={props.setShowSuggestions}
                                        setSelectedSuggestion={props.setSelectedSuggestion}
                                        localStorageStation={props.localStorageStation}/>
                ))}
        </div>
    )
}