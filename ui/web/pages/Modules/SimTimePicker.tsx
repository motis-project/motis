import React from "react";

export const SimTimePicker: React.FC = () => {
    return (
        <div className="sim-time-picker-container">
            <div className="sim-time-picker-overlay hidden">
                <div className="title"><input type="checkbox" id="sim-mode-checkbox" name="sim-mode-checkbox"><label
                        for="sim-mode-checkbox">Simulationsmodus</label></div>
                <div className="date disabled">
                    <div>
                        <div>
                            <div className="label">Datum</div>
                            <div className="gb-input-group">
                                <div className="gb-input-icon"><i className="icon">event</i></div><input className="gb-input"
                                    tabindex="20">
                                <div className="gb-input-widget">
                                    <div className="day-buttons">
                                        <div><a
                                                className="gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select"><i
                                                    className="icon">chevron_left</i></a></div>
                                        <div><a
                                                className="gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select"><i
                                                    className="icon">chevron_right</i></a></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="paper calendar hide">
                            <div className="month"><i className="icon">chevron_left</i><span className="month-name">Dezember
                                    2021</span><i className="icon">chevron_right</i></div>
                            <ul className="weekdays">
                                <li>Mo</li>11</li>
                                <li className="in-month invalid-day">12</li>
                                <li className="in-month invalid-day">13</li>
                                <li className="in-month invalid-day">14</li>
                                <li className="in-month invalid-day">15</li>
                                <li className="in-month invalid-day">16</li>
                                <li className="in-month invalid-day">17</li>
                                <li className="in-month invalid-day">18</li>
                                <li className="in-month invalid-day">19</li>
                                <li className="in-month invalid-day">20</li>
                                <li className="in-month invalid-day">21</li>
                                <li className="in-month invalid-day">22</li>
                                <li className="in-month invalid-day">23</li>
                                <li className="in-month invalid-day">24</li>
                                <li className="in-month invalid-day">25</li>
                                <li className="in-month invalid-day">26</li>
                                <li className="in-month invalid-day">27</li>
                                <li className="in-month invalid-day">28</li>
                                <li className="in-month invalid-day">29</li>
                                <li className="in-month invalid-day">30</li>
                                <li className="in-month invalid-day">31</li>
                                <li className="out-of-month invalid-day">1</li>
                                <li className="out-of-month invalid-day">2</li>
                                <li className="out-of-month invalid-day">3</li>
                                <li className="out-of-month invalid-day">4</li>
                                <li className="out-of-month invalid-day">5</li>
                                <li className="out-of-month invalid-day">6</li>
                                <li className="out-of-month invalid-day">7</li>
                                <li className="out-of-month invalid-day">8</li>
                                <li className="out-of-month invalid-day">9</li>
                            </ul>
                        </div>
                    </div>
                </div>
                <div className="time disabled">
                    <div>
                        <div className="label">Uhrzeit</div>
                        <div className="gb-input-group">
                            <div className="gb-input-icon"><i className="icon">schedule</i></div><input className="gb-input"
                                tabindex="21">
                            <div className="gb-input-widget">
                                <div className="hour-buttons">
                                    <div><a
                                            className="gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select"><i
                                                className="icon">chevron_left</i></a></div>
                                    <div><a
                                            className="gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select"><i
                                                className="icon">chevron_right</i></a></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div className="close"><i className="icon">close</i></div>
            </div>
    )
}