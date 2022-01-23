import React, { useEffect, useState } from "react";
import { Station } from "./ConnectionTypes";
import { Translations } from "./Localization";
import { Mode, Modepicker } from "./ModePicker";
import { StationSuggestion, AddressSuggestion } from "./Proposals";
import { Address, AddressSuggestionResponse, StationSuggestionResponse } from "./SuggestionTypes";


const fetchSuggestions = (input: string, setAddresses: React.Dispatch<React.SetStateAction<Address[]>>, setStations: React.Dispatch<React.SetStateAction<Station[]>>) => {
    let requestURL = 'https://europe.motis-project.de/?elm=AddressSuggestions';

    let body = {
        destination: { type: 'Module', target: '/address'},
        content_type: 'AddressRequest',
        content: { input: input }
    };

    fetch(requestURL, getPostRequest(body))
            .then(res => res.json())
            .then((res: AddressSuggestionResponse) => {
                console.log("Response came in");
                console.log(res);
                setAddresses(res.content.guesses);
            });

    requestURL = 'https://europe.motis-project.de/?elm=StationSuggestions';

    let body2 = {
        destination: { type: 'Module', target: '/guesser'},
        content_type: 'StationGuesserRequest',
        content: { guess_count: 6, input: input }
    };

    fetch(requestURL, getPostRequest(body2))
            .then(res => res.json())
            .then((res: StationSuggestionResponse) => {
                console.log("Response came in");
                console.log(res);
                setStations(res.content.guesses);
            });
};


const getPostRequest = (body: any) => {
    return {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify( body )
    };
};


export const SearchInputField: React.FC<{ 'translation': Translations, 'label': String, 'searchQuery': boolean, 'setSearchQuery': React.Dispatch<React.SetStateAction<boolean>>, 'station': Station | Address }> = (props) => {
    

    // StartType
    const [type, setType] = useState<string>('PretripStart');
    
    // StartModes
    const [modes, setModes] = useState<Mode[]>([]);
    
    // Start Station or Position
    const [station, setStation] = useState<Station | Address>(props.station);//<Start>({ station: { name: 'Darmstadt Hauptbahnhof', id: 'delfi_de:06411:4734:64:63'}, min_connection_count: 5, interval: { begin: 1640430180, end: 164043738 }, extend_interval_later: true, extend_interval_earlier: true });

    // StartName stores current input in the Start Input Field
    const [name, setName] = useState<string>(props.station.name);

    // Start Top Suggestion
    const [topSuggestion, setTopSuggestion] = useState<Station | Address>();
    
    // fetch Start Address Suggestions
    const [fetchSuggestionsFlag, setFetchSuggestionsFlag] = useState<boolean>(false);
    
    // Start Address Suggestions
    const [addressSuggestions, setAddressSuggestions] = useState<Address[]>([]);
    
    // Start Station Suggestions
    const [stationSuggestions, setStationSuggestions] = useState<Station[]>([]);
    
    // show Start Suggestions
    const [showSuggestions, setShowSuggestions] = useState<boolean>(false);
    
    
    //
    useEffect(() => {
        fetchSuggestions(name, setAddressSuggestions, setStationSuggestions);
        if (stationSuggestions.length == 0 ) {
            setTopSuggestion(addressSuggestions[0]);
        }else {
            setTopSuggestion(stationSuggestions[0]);
        }
        console.log(topSuggestion)
    }, [fetchSuggestionsFlag]);

    useEffect(() => {
        setName(props.station.name)
        setStation(props.station)
        setFetchSuggestionsFlag(!fetchSuggestionsFlag);
    }, [props.station])
    
    return (
        <div>
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
                                //console.log("Start changed")
                                setName(e.currentTarget.value)
                                if (e.currentTarget.value.length >= 3) {
                                    setFetchSuggestionsFlag(!fetchSuggestionsFlag);
                                }
                            } }
                            onKeyPress={e => {
                                if (e.key == "Enter") {
                                    e.preventDefault();
                                    console.log("Pressed Enter in Start")
                                    setName(stationSuggestions.length > 0 ? stationSuggestions[0].name : addressSuggestions.length > 0 ? addressSuggestions[0].name : '')
                                    setStation(stationSuggestions.length > 0 ? stationSuggestions[0] : addressSuggestions.length > 0 ? addressSuggestions[0] : undefined)
                                    //props.setSearchQuery(!props.searchQuery)
                                }
                            } }
                            onFocus={_ => {
                                setShowSuggestions(true);
                            } }/></div>
                </form>
                <div className='paper' style={showSuggestions && addressSuggestions.length > 0 ? {} : {display: 'none'}}>
                    <div className="proposals">
                        {
                            stationSuggestions.map((station: Station, index: number) => (
                                <StationSuggestion  station={station} 
                                                    key={index} 
                                                    setName={setName} 
                                                    setSuggestion={setStation} 
                                                    setShowSuggestions={setShowSuggestions}/>
                            ))}
                        {
                            addressSuggestions.map((address: Address, index: number) => (
                                <AddressSuggestion  address={address} 
                                                    key={index} 
                                                    setName={setName} 
                                                    setSuggestion={setStation} 
                                                    setShowSuggestions={setShowSuggestions}/>
                            ))}
                    </div>
                </div>
            </div>
            <Modepicker translation={props.translation} start={true}/>{/* modes={startModes} setModes={setStartModes}/>*/}
        </div>
    )
                    
}