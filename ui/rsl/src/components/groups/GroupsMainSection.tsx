import { Outlet } from "react-router-dom";

import GroupList from "@/components/groups/GroupList";

function GroupsMainSection(): JSX.Element {
  return (
    <>
      <div className="bg-db-cool-gray-200 dark:bg-gray-800 w-[25rem] overflow-y-auto p-2 shrink-0">
        <GroupList />
      </div>
      <div className="overflow-y-auto grow p-2">
        <Outlet />
      </div>
    </>
  );
}

export default GroupsMainSection;
