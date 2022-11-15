import { Link } from "react-router-dom";

import getQueryParameters from "@/util/queryParameters";

import MainMenu from "@/components/header/MainMenu";
import Settings from "@/components/header/Settings";
import TimeControl from "@/components/header/TimeControl";
import UniverseControl from "@/components/header/UniverseControl";

import logoUrl from "@/logo.svg";

const allowForwarding = getQueryParameters().get("allowForwarding") === "yes";

function Header(): JSX.Element {
  return (
    <div
      className="flex justify-between items-center p-2
            bg-db-cool-gray-200 dark:bg-gray-800 text-black dark:text-neutral-300
            border-b-2 border-db-cool-gray-600"
    >
      <div className="flex items-center space-x-4">
        <Link to="/">
          <img src={logoUrl} className="h-8 w-auto" alt="MOTIS RSL" />
        </Link>
        <MainMenu />
        <TimeControl allowForwarding={allowForwarding} />
      </div>

      <div className="flex items-center space-x-8">
        <UniverseControl />
        <Settings />
      </div>
    </div>
  );
}

export default Header;
