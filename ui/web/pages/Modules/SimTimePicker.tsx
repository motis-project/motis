import React from "react";

import { DatePicker } from "./DatePicker";

export const SimTimePicker: React.FC = (props) => {
    return (
        <div className="sim-time-picker-container">
            <div className="sim-time-picker-overlay">
                <div className="title">
                    <input type="checkbox" id="sim-mode-checkbox" name="sim-mode-checkbox" />
                    <label htmlFor="sim-mode-checkbox">Simulationsmodus</label>
                </div>
                <div className="date">
                    <div>
                        <div>
                            <div className="label">Datum</div>
                            <div className="gb-input-group">
                                <div className="gb-input-icon">
                                    <i className="icon">event</i>
                                </div>
                                <input className="gb-input" tabIndex={20} />
                                <div className="gb-input-widget">
                                    <div className="day-buttons">
                                        <div>
                                            <a className="gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select">
                                                <i className="icon">chevron_left</i>
                                            </a>
                                        </div>
                                        <div>
                                            <a className="gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select">
                                                <i className="icon">chevron_right</i>
                                            </a>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <DatePicker />
                    </div>
                </div>
                <div className="time">
                    <div>
                        <div className="label">Uhrzeit</div>
                        <div className="gb-input-group">
                            <div className="gb-input-icon"><i className="icon">schedule</i></div>
                            <input className="gb-input" tabIndex={21} />
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
                <div className="close">
                    <i className="icon">close</i>
                </div>
            </div>
        </div>
    )
}