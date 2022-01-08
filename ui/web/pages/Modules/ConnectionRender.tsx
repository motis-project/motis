import React from 'react';

export const ConnectionRender: React.FC = (connectionArr) => {
    return (
        <div className="connection">
            <div className="pure-g">
                <div className="pure-u-4-24 connection-times">
                    <div className="connection-departure">14:20 </div>
                    <div className="connection-arrival">14:35 </div>
                </div>
                <div className="pure-u-4-24 connection-duration">
                    <div>15min</div>
                </div>
                <div className="pure-u-16-24 connection-trains">
                    <div className="transport-graph"><svg width="335" height="40" viewBox="0 0 335 40">
                        <g>
                            <g className="part train-className-2 acc-0">
                                <line x1="0" y1="12" x2="326" y2="12" className="train-line"></line>
                                <circle cx="12" cy="12" r="12" className="train-circle"></circle>
                                <use xlinkHref="#train" className="train-icon" x="4" y="4" width="16" height="16"></use><text
                                    x="0" y="40" text-anchor="start" className="train-name">IC 117</text>
                                <rect x="0" y="0" width="323" height="24" className="tooltipTrigger"></rect>
                            </g>
                        </g>
                        <g className="destination">
                            <circle cx="329" cy="12" r="6"></circle>
                        </g>
                    </svg>
                        <div className="tooltip" style={{position: "absolute", left: "0px", top: "23px"}}>
                            <div className="stations">
                                <div className="departure"><span className="station">Frankfurt (Main) Hauptbahnhof</span><span
                                    className="time">14:20</span></div>
                                <div className="arrival"><span className="station">Darmstadt Hauptbahnhof</span><span
                                    className="time">14:35</span></div>
                            </div>
                            <div className="transport-name"><span>IC 117</span></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};