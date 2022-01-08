import React from "react";

export const StationSearch: React.FC = (props) => {

    return(
        <div id="station-search" className="">
            <div>
                <div>
                    <div className="label"></div>
                    <div className="gb-input-group">
                        <div className="gb-input-icon">
                            <i className="icon">place</i>
                        </div>
                        <input className="gb-input" tabIndex={10}/>
                    </div>
                </div>
                <div>
                    <div className="paper hide">
                        <ul className="proposals"></ul>
                    </div>
                </div>
            </div>
        </div>
    );
};