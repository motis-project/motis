// import App from 'next/app'

import '../style/app.css';
import '../style/button.css';
import '../style/calendar.css';
import '../style/global.css';
import '../style/grids-gutters.css';
import '../style/grids-min.css';
import '../style/grids-responsive-min.css';
import '../style/input.css';
import '../style/mapbox-gl.css'
//import '../style/search-form.css';
import '../style/spinner.css';
import '../style/taglist.css';
import '../style/time.css';
import '../style/transport-graph.css';
import '../style/typeahead.css';

function MyApp({ Component, pageProps }) {
    return <Component {...pageProps} />
  }
  
  // Only uncomment this method if you have blocking data requirements for
  // every single page in your application. This disables the ability to
  // perform automatic static optimization, causing every page in your app to
  // be server-side rendered.
  //
  // MyApp.getInitialProps = async (appContext) => {
  //   // calls page's `getInitialProps` and fills `appProps.pageProps`
  //   const appProps = await App.getInitialProps(appContext);
  //
  //   return { ...appProps }
  // }
  
  export default MyApp