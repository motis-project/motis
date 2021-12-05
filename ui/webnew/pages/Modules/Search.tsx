import React from 'react';

//import DatePicker from 'sassy-datepicker';

export const Search: React.FC = () => {
    return (
        <div className="search">
            <div className="pure-g gutters">
                <div className="pure-u-1">
                    <div>
                        <div>
                            <div className="label">Start</div>
                            <div className="gb-input-group">
                                <div className="gb-input-icon">{/*<img className="icon">place</img>*/}</div><input
                                    className="gb-input" tabIndex={1} />
                            </div>
                        </div>
                        <div className="paper">
                            <ul className="proposals"></ul>
                        </div>
                    </div>
                    <div>
                        {/**<ModePicker /> */}
                    </div>
                    <div className="swap-locations-btn">
                        <label className="gb-button">
                            <input type="checkbox" />
                            {/*<img className="icon">swap_vert</img>*/}
                        </label>
                    </div>
                </div>
                <div className="pure-u-1">
                    <div>
                        <div>
                            <div className="label">Datum</div>
                            {/*<DatePicker />*/}
                        </div>
                    </div>
                </div>
            </div>
            <div className="pure-g gutters">
                <div className="pure-u-1">

                </div>
                <div className="pure-u-1">

                </div>
                <div className="pure-u-1">

                </div>
            </div>
        </div>
    )
}