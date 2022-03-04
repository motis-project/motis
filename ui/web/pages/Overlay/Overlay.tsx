import React, { useState } from 'react';

import { Search } from './Search';
import { SubOverlay } from './SubOverlay';
import { Connection, Station, Transport, TripId } from '../Types/ConnectionTypes';
import { Translations } from '../App/Localization';
import { ConnectionRender, JourneyRender, duration } from './ConnectionRender';
import { getFromLocalStorage } from '../App/LocalStorage';
import { Address } from '../Types/SuggestionTypes';
import moment from 'moment';

const getTransportCountString = (transports: Transport[], translation: Translations) => {
    let count = 0;
    for (let index = 0; index < transports.length; index++) {
        if (transports[index].move_type === 'Transport' && index > 0) {
            count++
        }
    }
    return translation.connections.interchanges(count);
}

export const Overlay: React.FC<{ 'translation': Translations }> = (props) => {

    // Hold the currently displayed Date
    const [displayDate, setDisplayDate] =useState<moment.Moment>(moment());
    
    // Boolean used to decide if the Overlay is being displayed
    const [overlayHidden, setOverlayHidden] = useState<Boolean>(true);

    // Boolean used to decide if the SubOverlay is being displayed
    const [subOverlayHidden, setSubOverlayHidden] = useState<Boolean>(true);

    // Connections
    const [connections, setConnections] = useState<Connection[]>(null);

    // Boolean used to signal <Search> that extendForward was clicked
    const [extendForwardFlag, setExtendForwardFlag] = useState<boolean>(true);

    // Boolean used to signal <Search> that extendBackward was clicked
    const [extendBackwardFlag, setExtendBackwardFlag] = useState<boolean>(true);
    
    const [detailViewHidden, setDetailViewHidden] = useState<Boolean>(true);

    const [indexOfConnection, setIndexOfConnection] = useState<number>(0);

    const [trainSelected, setTrainSelected] = useState<TripId>(undefined);
    
    const [start, setStart] = useState<Station | Address>(getFromLocalStorage("motis.routing.from_location"));

    const [destination, setDestination] = useState<Station | Address>(getFromLocalStorage("motis.routing.to_location"));

    return (
        <div className={overlayHidden ? 'overlay-container' : 'overlay-container hidden'}>
            <div className='overlay'>
                <div id='overlay-content'>
                    {detailViewHidden ?
                        <>
                            <Search setConnections={setConnections} 
                                    translation={props.translation} 
                                    extendForwardFlag={extendForwardFlag}
                                    extendBackwardFlag={extendBackwardFlag}
                                    displayDate={displayDate}
                                    setDisplayDate={setDisplayDate}/>
                            {!connections ? 
                                <div className='spinner'>
                                    <div className='bounce1'></div>
                                    <div className='bounce2'></div>
                                    <div className='bounce3'></div>
                                </div> 
                                : 
                                <div id='connections'>
                                    <div className='connections'>
                                        <div className='extend-search-interval search-before' onClick={() => setExtendBackwardFlag(!extendBackwardFlag)}><a>{props.translation.connections.extendBefore}</a></div>
                                        <div className='connection-list'>
                                            {connections.map((connectionElem: Connection, index) => (
                                                connectionElem.dummyDay ?
                                                <div className='date-header divider' key={index}><span>{connectionElem.dummyDay}</span></div>
                                                :
                                                <div className='connection' key={index} onClick={() => { setDetailViewHidden(false); setIndexOfConnection(index) }}>
                                                    <div className='pure-g'>
                                                        <div className='pure-u-4-24 connection-times'>
                                                            <div className='connection-departure'>
                                                                {moment.unix(connectionElem.stops[0].departure.time).format('HH:mm')}
                                                            </div>
                                                            <div className='connection-arrival'>
                                                                {moment.unix(connectionElem.stops[connectionElem.stops.length - 1].arrival.time).format('HH:mm')}
                                                            </div>
                                                        </div>
                                                        <div className='pure-u-4-24 connection-duration'>
                                                            {duration(connectionElem.stops[0].departure.time, connectionElem.stops[connectionElem.stops.length - 1].arrival.time)}
                                                        </div>
                                                        <div className='pure-u-16-24 connection-trains'>
                                                            <div className='transport-graph'>
                                                                <ConnectionRender connection={connectionElem} setDetailViewHidden={setDetailViewHidden}/>
                                                                {/* Was ist tooltip? Es ist unsichtbar und hat keine Funktion meiner Meinung nach.*/ }

                                                                <div className='tooltip' style={{ position: 'absolute', left: '0px', top: '23px' }}>
                                                                    <div className='stations'>
                                                                        <div className='departure'><span className='station'>{props.translation.search.departure}</span><span
                                                                            className='time'>14:20</span></div>
                                                                        <div className='arrival'><span className='station'>{props.translation.search.arrival}</span><span
                                                                            className='time'>14:35</span></div>
                                                                    </div>
                                                                    <div className='transport-name'><span>IC 117</span></div>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            ))}
                                            <div className='divider footer'></div>
                                            <div className='extend-search-interval search-after' onClick={() => setExtendForwardFlag(!extendForwardFlag)}><a>{props.translation.connections.extendAfter}</a></div>
                                        </div>
                                    </div>
                                </div>
                            }
                        </> :
                        <div className="connection-details">
                            <div className="connection-info">
                                <div className="header">
                                    <div className="back"><i className="icon" onClick={() => setDetailViewHidden(true)}>arrow_back</i></div>
                                    <div className="details">
                                        <div className="date">{displayDate.unix()}</div>
                                        <div className="connection-times">
                                            <div className="times">
                                                <div className="connection-departure">{moment.unix(connections[indexOfConnection].stops[0].departure.time).format('HH:mm')}</div>
                                                <div className="connection-arrival">{moment.unix(connections[indexOfConnection].stops[connections[indexOfConnection].stops.length - 1].arrival.time).format('HH:mm')}</div>
                                            </div>
                                            <div className="locations">
                                                <div>{start.name}</div>
                                                <div>{destination.name}</div>
                                            </div>
                                        </div>
                                        <div className="summary">
                                            <span className="duration">
                                                <i className="icon">schedule</i>
                                                {duration(connections[indexOfConnection].stops[0].departure.time, connections[indexOfConnection].stops[connections[indexOfConnection].stops.length - 1].arrival.time)}
                                            </span>
                                            <span className="interchanges">
                                                <i className="icon">transfer_within_a_station</i>
                                                {getTransportCountString(connections[indexOfConnection].transports, props.translation)}
                                            </span>
                                        </div>
                                    </div>
                                    <div className="actions"><i className="icon">save</i><i className="icon">share</i></div>
                                </div>
                            </div>
                            <div className="connection-journey" id="connection-journey">
                                <JourneyRender connection={connections[indexOfConnection]} setSubOverlayHidden={setSubOverlayHidden} setTrainSelected={setTrainSelected} detailViewHidden={detailViewHidden} translation={props.translation}/>
                            </div>
                        </div>
                    }
                </div>
                <SubOverlay subOverlayHidden={subOverlayHidden} setSubOverlayHidden={setSubOverlayHidden} trainSelected={trainSelected} setTrainSelected={setTrainSelected} translation={props.translation} detailViewHidden={detailViewHidden}/>
            </div>
            <div className='overlay-tabs'>
                <div className='overlay-toggle' onClick={() => setOverlayHidden(!overlayHidden)}>
                    <i className='icon'>arrow_drop_down</i>
                </div>
                <div className={subOverlayHidden ? 'trip-search-toggle' : 'trip-search-toggle enabled'} onClick={() => {setSubOverlayHidden(!subOverlayHidden), setTrainSelected(undefined)}}>
                    <i className='icon'>train</i>
                </div>
            </div>
        </div>
    );
};