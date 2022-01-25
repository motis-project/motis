import React from 'react';

import { Station } from './ConnectionTypes';
import { Address, Region } from './SuggestionTypes';


const getDisplayRegions = (regions: Region[]) => {
    let res = regions.filter((r: Region) => r.admin_level <= 8 && r.admin_level >= 2);
    return res[0].name + ', ' + res[res.length - 1].name;
}


function useOutsideAlerter(ref: React.MutableRefObject<any>, showSuggestions: boolean, setShowSuggestions: React.Dispatch<React.SetStateAction<boolean>>) {
    React.useEffect(() => {
        /**
         * Alert if clicked on outside of element
         */
        function handleClickOutside(event) {
            if (ref.current && !ref.current.contains(event.target)) {
                setShowSuggestions(false);
                console.log('Clicked outside');
            }
        }

        // Bind the event listener
        console.log(showSuggestions);
        if (showSuggestions){
            document.addEventListener("mousedown", handleClickOutside);
        }
        return () => {
            // Unbind the event listener on clean up
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, [ref]);
}


const StationSuggestion: React.FC<{'station': Station, 'setName': React.Dispatch<React.SetStateAction<string>>, 'setSuggestion': React.Dispatch<React.SetStateAction<Station | Address>>, 'setShowSuggestions': React.Dispatch<React.SetStateAction<boolean>>, 'listID': number, 'selectedSuggestion': number, 'setSelectedSuggestion': React.Dispatch<React.SetStateAction<number>>}> = (props) => {

    return (
        <li className={props.listID == props.selectedSuggestion ? 'selected' : ''} 
            title={props.station.name} 
            onMouseDown={(e) => {
                if(e.button == 0){
                    props.setName(props.station.name); 
                    props.setSuggestion(props.station); 
                    props.setShowSuggestions(false);
                    props.setSelectedSuggestion(0);
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


const AddressSuggestion: React.FC<{'address': Address, 'setName': React.Dispatch<React.SetStateAction<string>>, 'setSuggestion': React.Dispatch<React.SetStateAction<Station | Address>>, 'setShowSuggestions': React.Dispatch<React.SetStateAction<boolean>>, 'listID': number, 'selectedSuggestion': number, 'setSelectedSuggestion': React.Dispatch<React.SetStateAction<number>>}> = (props) => {

    return (
        <li className={props.listID == props.selectedSuggestion ? 'selected' : ''}
            title={props.address.name + ', ' + getDisplayRegions(props.address.regions).split(',')[0]} 
            onMouseDown={() => {
                props.setName(props.address.name); 
                props.setSuggestion(props.address);
                props.setShowSuggestions(false);
                props.setSelectedSuggestion(0) 
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


export const Proposals: React.FC<{'addresses': Address[], 'stations': Station[], 'suggestions': (Address | Station) [], 'highlighted': number, 'showSuggestions': boolean, 'setName': React.Dispatch<React.SetStateAction<string>>, 'setSuggestion': React.Dispatch<React.SetStateAction<Station | Address>>, 'setShowSuggestions': React.Dispatch<React.SetStateAction<boolean>>, 'setHighlighted': React.Dispatch<React.SetStateAction<number>>}> = (props) => {

    const suggestionsRef = React.useRef(null);
    useOutsideAlerter(suggestionsRef, props.showSuggestions, props.setShowSuggestions);

    /*React.useEffect(() => {
        const pageClickEvent = (e) => {
            console.log('Clicked anywhere');
            console.log(suggestionsRef.current);
            console.log(suggestionsRef.current !== null);
            console.log(!suggestionsRef.current?.contains(e.target))
            console.log(e.target)
            if (suggestionsRef.current !== null && !suggestionsRef.current?.contains(e.target)) {
                props.setShowSuggestions(false);
                console.log('Clicked outside');
            }
        };

        if (props.showSuggestions) {
            window.addEventListener('click', pageClickEvent);
        };

        return () => {
            window.removeEventListener('click', pageClickEvent);
          }

    }, [props.showSuggestions])*/

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
                                        setSelectedSuggestion={props.setHighlighted}/>
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
                                        setSelectedSuggestion={props.setHighlighted}/>
                ))}
        </div>
    )
}