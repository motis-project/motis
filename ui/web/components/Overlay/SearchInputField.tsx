import React, { useEffect, useState } from 'react';
import { Station } from '../Types/Connection';
import { Translations } from '../App/Localization';
import { Proposals } from './Proposals';
import { Address, AddressSuggestionResponse, StationSuggestionResponse } from '../Types/SuggestionTypes';
import { setLocalStorage } from '../App/LocalStorage';


interface SearchInputField {
    translation: Translations, 
    label: String, 
    station: Station | Address, 
    localStorageStation: string
    setSearchDisplay: React.Dispatch<React.SetStateAction<Station | Address>>, 
}


const fetchSuggestions = (input: string, setAddresses: React.Dispatch<React.SetStateAction<Address[]>>, setStations: React.Dispatch<React.SetStateAction<Station[]>>, setSuggestions: React.Dispatch<React.SetStateAction<(Station | Address)[]>>) => {
    if (input !== '') {
        let requestURL = 'https://europe.motis-project.de/?elm=StationSuggestions';
        
        let body = {
            destination: { type: 'Module', target: '/guesser'},
            content_type: 'StationGuesserRequest',
            content: { guess_count: 6, input: input }
        };

        let stationRes = [];
        
        fetch(requestURL, getPostRequest(body))
        .then(res => res.json())
        .then((res: StationSuggestionResponse) => {
            console.log('Response came in');
            console.log(res);
            setStations(res.content.guesses);
            stationRes = res.content.guesses;
        });

        requestURL = 'https://europe.motis-project.de/?elm=AddressSuggestions';
        
        let body2 = {
            destination: { type: 'Module', target: '/address'},
            content_type: 'AddressRequest',
            content: { input: input }
        };

        fetch(requestURL, getPostRequest(body2))
                .then(res => res.json())
                .then((res: AddressSuggestionResponse) => {
                    console.log('Response came in');
                    console.log(res);
                    setAddresses(res.content.guesses);
                    setSuggestions([...stationRes, ...res.content.guesses]);
                });
        }
};


const getPostRequest = (body: any) => {
    return {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify( body )
    };
};


export const SearchInputField: React.FC<SearchInputField> = (props) => {
    
    const inputFieldRef = React.useRef(null);
    
    // Selected manipulates the div "gb-input-group" to highlight it if focused
    const [selected, setSelected] = useState<string>('');

    // Station or Position
    const [station, setStation] = useState<Station | Address>(props.station);

    // Name stores current input in the Input Field
    const [name, setName] = useState<string>('');
    
    // fetch Address Suggestions
    const [fetchSuggestionsFlag, setFetchSuggestionsFlag] = useState<boolean>(false);
    
    // Address Suggestions
    const [addressSuggestions, setAddressSuggestions] = useState<Address[]>([]);
    
    // Station Suggestions
    const [stationSuggestions, setStationSuggestions] = useState<Station[]>([]);

    // List of all Suggestions
    const [suggestions, setSuggestions] = useState<(Address | Station)[]>([]);
    
    // show Suggestions
    const [showSuggestions, setShowSuggestions] = useState<boolean>(false);

    // Index, tracking which suggestion is highlighted right now
    const [selectedSuggestion, setSelectedSuggestion] = React.useState<number>(0);

    //
    useEffect(() => {
        fetchSuggestions(name, setAddressSuggestions, setStationSuggestions, setSuggestions);
    }, [fetchSuggestionsFlag]);

    useEffect(() => {
        setName(props.station == null ? '' : props.station.name)
        setStation(props.station)
    }, [props.station])

    useEffect(() => {
        setFetchSuggestionsFlag(!fetchSuggestionsFlag);
        props.setSearchDisplay(station);
    }, [station])
    
    return (
        <div>
            <form>
                <div className='label'>
                    {props.label}
                </div>
                <div className={`gb-input-group ${selected}`}>
                    <div className='gb-input-icon'>
                        <i className='icon'>place</i>
                        </div>
                <input  className='gb-input' tabIndex={1} value={name} ref={inputFieldRef} 
                        onChange={e => {
                            //e.preventDefault();
                            setName(e.currentTarget.value)
                            if (e.currentTarget.value.length >= 3) {
                                setFetchSuggestionsFlag(!fetchSuggestionsFlag);
                                setShowSuggestions(true);
                            }
                        } }
                        onKeyDown={e => {
                            switch (e.key) {
                                case 'Enter':
                                    e.preventDefault();
                                    setName(suggestions[selectedSuggestion].name);
                                    setStation(suggestions[selectedSuggestion]);
                                    setShowSuggestions(false);
                                    setSelectedSuggestion(0);
                                    setLocalStorage(props.localStorageStation, suggestions[selectedSuggestion]);
                                    break;
                                case 'Escape':
                                    e.preventDefault();
                                    setShowSuggestions(false);
                                    break;
                                case 'ArrowDown':
                                    setSelectedSuggestion(selectedSuggestion + 1);
                                    break;
                                case 'ArrowUp':
                                    setSelectedSuggestion(selectedSuggestion - 1);
                                    break;
                                default:
                                }
                        } }
                        onFocus={_ => {
                            setShowSuggestions(true);
                            setSelected('gb-input-group-selected');
                        } }
                        onClick={_ => {
                            setShowSuggestions(true);
                        } }/></div>
            </form>
            <div className='paper' style={showSuggestions && addressSuggestions.length > 0 ? {} : {display: 'none'}}>
                <Proposals  addresses={addressSuggestions} 
                            stations={stationSuggestions}
                            suggestions={suggestions}
                            highlighted={selectedSuggestion}
                            showSuggestions={showSuggestions}
                            setName={setName}
                            setSuggestion={setStation} 
                            setSelectedSuggestion={setSelectedSuggestion}
                            setShowSuggestions={setShowSuggestions}
                            setSelected={setSelected}
                            localStorageStation={props.localStorageStation}
                            inputFieldRef={inputFieldRef}/>
            </div>
        </div>
    )
                    
}