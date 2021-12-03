import React from 'react';
import ReactDOM from 'react-dom'

import { Main } from './App/App'

class Index extends React.Component{
    state = {};
    render() {
        return <React.StrictMode>
                    <Main />
            </React.StrictMode>;
    }
}

/*if(typeof document !== 'undefined'){
    ReactDOM.render(
        <React.StrictMode>
            <Main />
        </React.StrictMode>,
        document.getElementById('root')
    );
}*/

export default Index;