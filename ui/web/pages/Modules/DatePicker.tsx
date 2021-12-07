import React from "react";

export const DatePicker: React.FC = () => {
    
    const[datePickerSelected, setDatePickerSelected] = React.useState<Boolean>(false);
    
    return (
        <div>
            <div>
                <div className="label">Datum</div>
                <div className="gb-input-group">
                    <div className="gb-input-icon">
                        <i className="icon">event</i></div>
                    <input className="gb-input" tabIndex={3} onBlur={() => setDatePickerSelected(false)} onFocus={() => setDatePickerSelected(true)}/>
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
            <div className={datePickerSelected ? "paper calendar" : "paper calendar hide"}>
                <div className="month">
                    <i className="icon">chevron_left</i>
                    <span className="month-name">Oktober 2020</span>
                    <i className="icon">chevron_right</i>
                </div>
                <ul className="weekdays">
                    <li>Mo</li>
                    <li>Di</li>
                    <li>Mi</li>
                    <li>Do</li>
                    <li>Fr</li>
                    <li>Sa</li>
                    <li>So</li>
                </ul>
                <ul className="calendardays">
                    <li className="out-of-month invalid-day">28</li>
                    <li className="out-of-month invalid-day">29</li>
                    <li className="out-of-month invalid-day">30</li>
                    <li className="in-month invalid-day">1</li>
                    <li className="in-month invalid-day">2</li>
                    <li className="in-month invalid-day">3</li>
                    <li className="in-month invalid-day">4</li>
                    <li className="in-month invalid-day">5</li>
                    <li className="in-month invalid-day">6</li>
                    <li className="in-month invalid-day">7</li>
                    <li className="in-month invalid-day">8</li>
                    <li className="in-month invalid-day">9</li>
                    <li className="in-month invalid-day">10</li>
                    <li className="in-month invalid-day">11</li>
                    <li className="in-month invalid-day">12</li>
                    <li className="in-month invalid-day">13</li>
                    <li className="in-month invalid-day">14</li>
                    <li className="in-month invalid-day">15</li>
                    <li className="in-month invalid-day">16</li>
                    <li className="in-month invalid-day">17</li>
                    <li className="in-month invalid-day">18</li>
                    <li className="in-month today valid-day">19</li>
                    <li className="in-month selected valid-day">20</li>
                    <li className="in-month valid-day">21</li>
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
                </ul>
            </div>
        </div>
    )
}