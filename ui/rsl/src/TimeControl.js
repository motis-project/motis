import { formatDateTime } from "./util/dateFormat";

function TimeControl(props) {
  return (
    <div className="flex flex-row items-center space-x-2 m-2">
      {props.systemTime ? (
        <>
          <div>System time: {formatDateTime(props.systemTime)}</div>
          <button
            className="bg-gray-200 px-2 py-1 border border-gray-300 rounded-xl"
            disabled={props.disabled}
            onClick={() => props.onForwardTime(props.systemTime + 60)}
          >
            +1m
          </button>
          {[1, 5, 10].map((hrs) => (
            <button
              key={hrs.toString()}
              className="bg-gray-200 px-2 py-1 border border-gray-300 rounded-xl"
              disabled={props.disabled}
              onClick={() =>
                props.onForwardTime(props.systemTime + 60 * 60 * hrs)
              }
            >
              +{hrs}h
            </button>
          ))}
        </>
      ) : (
        <div>System time: loading...</div>
      )}
    </div>
  );
}

export default TimeControl;
