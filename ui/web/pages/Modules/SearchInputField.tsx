import React, { useEffect, useState } from 'react';
import { Station } from './ConnectionTypes';
import { Translations } from './Localization';
import { Modepicker } from './ModePicker';
import { Mode } from './IntermodalRoutingTypes';
import { Proposals } from './Proposals';
import { Address, AddressSuggestionResponse, StationSuggestionResponse } from './SuggestionTypes';
import { setLocalStorage } from './LocalStorage';


const fetchSuggestions = (input: string, setAddresses: React.Dispatch<React.SetStateAction<Address[]>>, setStations: React.Dispatch<React.SetStateAction<Station[]>>, setSuggestions: React.Dispatch<React.SetStateAction<(Station | Address)[]>>) => {
    if (input !== '') {
        console.log(input);
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


export const SearchInputField: React.FC<{ 'translation': Translations, 'label': String, 'searchQuery': boolean, 'setSearchQuery': React.Dispatch<React.SetStateAction<boolean>>, 'station': Station | Address, 'setSearchDisplay': React.Dispatch<React.SetStateAction<Station | Address>>, 'localStorageStation': string }> = (props) => {
    

    // Type
    const [type, setType] = useState<string>('PretripStart');
    
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
        //setFetchSuggestionsFlag(!fetchSuggestionsFlag);
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
                <div className='gb-input-group'>
                    <div className='gb-input-icon'>
                        <i className='icon'>place</i>
                        </div>
                <input  className='gb-input' tabIndex={1} value={name} 
                        onChange={e => {
                            //e.preventDefault();
                            setName(e.currentTarget.value)
                            if (e.currentTarget.value.length >= 3) {
                                setFetchSuggestionsFlag(!fetchSuggestionsFlag);
                                setShowSuggestions(true);
                            }
                        } }
                        onClick={_ => {
                            setShowSuggestions(true);
                        }}
                        onKeyDown={e => {
                            switch (e.key) {
                                case 'Enter':
                                    e.preventDefault();
                                    setName(suggestions[selectedSuggestion].name);
                                    setStation(suggestions[selectedSuggestion]);
                                    //props.setSearchDisplay(suggestions[selectedSuggestion]);
                                    setShowSuggestions(false);
                                    setSelectedSuggestion(0);
                                    //props.setSearchQuery(!props.searchQuery)
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
                                    console.log(e.key);
                                }
                        } }
                        onFocus={_ => {
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
                            setShowSuggestions={setShowSuggestions}
                            setHighlighted={setSelectedSuggestion}
                            localStorageStation={props.localStorageStation}/>
            </div>
        </div>
    )
                    
}