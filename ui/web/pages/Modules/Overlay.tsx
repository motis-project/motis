import React, { useState } from 'react';

import { Search } from './Search';
import { SubOverlay } from './SubOverlay';
import { Connection, Transport, TripId } from './ConnectionTypes';
import { Translations } from './Localization';
import { ConnectionRender, JourneyRender } from './ConnectionRender';


const displayTime = (posixTime) => {
    let today = new Date(posixTime * 1000);
    let h = String(today.getHours());
    let m = String(today.getMinutes()).padStart(2, '0');
    return h + ':' + m;
}

const displayDuration = (posixTime) => {
    let today = new Date(posixTime * 1000);
    let h = String(today.getUTCHours());
    let m = String(today.getUTCMinutes()).padStart(2, '0');
    if (h === '0') {
        return m + 'min';
    } else {
        return h + 'h ' + m + 'min';
    }
}

const getTransportCountString = (transports: Transport[]) => {
    let count = 0;
    for (let index = 0; index < transports.length; index++) {
        if (transports[index].move_type === 'Transport') {
            count++
        }
    }
    return count + ' Umstiege';
}

export const Overlay: React.FC<{ 'translation': Translations }> = (props) => {

    // Boolean used to decide if the Overlay is being displayed
    const [overlayHidden, setOverlayHidden] = useState<Boolean>(true);

    // Boolean used to decide if the SubOverlay is being displayed
    const [subOverlayHidden, setSubOverlayHidden] = useState<Boolean>(true);

    // Connections
    const [connections, setConnections] = useState<Connection[]>(null);

    const [detailViewHidden, setDetailViewHidden] = useState<Boolean>(true);

    const [indexOfConnection, setIndexOfConnection] = useState<number>(0);

    const [trainSelected, setTrainSelected] = useState<TripId>(undefined);

    return (
        <div className={overlayHidden ? 'overlay-container' : 'overlay-container hidden'}>
            <div className='overlay'>
                <div id='overlay-content'>
                    {detailViewHidden ?
                        <>
                            <Search setConnections={setConnections} translation={props.translation} />
                            {!connections ?
                                <div className='spinner'>
                                    <div className='bounce1'></div>
                                    <div className='bounce2'></div>
                                    <div className='bounce3'></div>
                                </div> :
                                <div id='connections'>
                                    {connections.map((connectionElem: Connection, index) => (
                                        <div className='connection' key={index} onClick={() => { setDetailViewHidden(false); setIndexOfConnection(index) }}>
                                            <div className='pure-g'>
                                                <div className='pure-u-4-24 connection-times'>
                                                    <div className='connection-departure'>
                                                        {displayTime(connectionElem.stops[0].departure.time)}
                                                    </div>
                                                    <div className='connection-arrivial'>
                                                        {displayTime(connectionElem.stops[connectionElem.stops.length - 1].arrival.time)}
                                                    </div>
                                                </div>
                                                <div className='pure-u-4-24 connection-duration'>
                                                    {displayDuration(new Date(connectionElem.stops[connectionElem.stops.length - 1].arrival.time).getTime() - new Date(connectionElem.stops[0].departure.time).getTime())}
                                                </div>
                                                <div className='pure-u-16-24 connection-trains'>
                                                    <div className='transport-graph'>
                                                        <ConnectionRender connection={connectionElem} setDetailViewHidden={setDetailViewHidden} />
                                                        <div className='tooltip' style={{ position: 'absolute', left: '0px', top: '23px' }}>
                                                            <div className='stations'>
                                                                <div className='departure'><span className='station'>Frankfurt (Main) Hauptbahnhof</span><span
                                                                    className='time'>14:20</span></div>
                                                                <div className='arrival'><span className='station'>Darmstadt Hauptbahnhof</span><span
                                                                    className='time'>14:35</span></div>
                                                            </div>
                                                            <div className='transport-name'><span>IC 117</span></div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>}
                        </> :
                        <div className="connection-details">
                            <div className="connection-info">
                                <div className="header">
                                    <div className="back"><i className="icon" onClick={() => setDetailViewHidden(true)}>arrow_back</i></div>
                                    <div className="details">
                                        <div className="date">24.1.2022</div>
                                        <div className="connection-times">
                                            <div className="times">
                                                <div className="connection-departure">{displayTime(connections[indexOfConnection].stops[0].departure.time)}</div>
                                                <div className="connection-arrival">{displayTime(connections[indexOfConnection].stops[connections[indexOfConnection].stops.length - 1].arrival.time)}</div>
                                            </div>
                                            <div className="locations">
                                                <div>{connections[indexOfConnection].stops[0].station.name}</div>
                                                <div>{connections[indexOfConnection].stops[connections[indexOfConnection].stops.length - 1].station.name}</div>
                                            </div>
                                        </div>
                                        <div className="summary">
                                            <span className="duration">
                                                <i className="icon">schedule</i>
                                                {displayDuration(new Date(connections[indexOfConnection].stops[connections[indexOfConnection].stops.length - 1].arrival.time).getTime() - new Date(connections[indexOfConnection].stops[0].departure.time).getTime())}
                                            </span>
                                            <span className="interchanges">
                                                <i className="icon">transfer_within_a_station</i>
                                                {getTransportCountString(connections[indexOfConnection].transports)}
                                            </span>
                                        </div>
                                    </div>
                                    <div className="actions"><i className="icon">save</i><i className="icon">share</i></div>
                                </div>
                            </div>
                            <div className="connection-journey" id="connection-journey">
                                <JourneyRender connection={connections[indexOfConnection]} setSubOverlayHidden={setSubOverlayHidden} setTrainSelected={setTrainSelected} />
                            </div>
                        </div>}
                </div>
                <SubOverlay subOverlayHidden={subOverlayHidden} setSubOverlayHidden={setSubOverlayHidden} trainSelected={trainSelected} setTrainSelected={setTrainSelected} translation={props.translation} />
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