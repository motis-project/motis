import { useAtom } from "jotai";
import { ReactNode } from "react";

import { MainPage, mainPageAtom, showSimPanelAtom } from "@/data/views";

import classNames from "@/util/classNames";

type PageLinkProps = {
  active?: boolean;
  onClick?: () => void;
  children: ReactNode;
};

function PageLink({ active, onClick, children }: PageLinkProps): JSX.Element {
  return (
    <div
      className={classNames(
        "px-3 py-2 rounded-md text-sm font-medium cursor-pointer",
        active
          ? "bg-db-cool-gray-700 text-white"
          : "hover:bg-db-cool-gray-300 text-black"
      )}
      onClick={onClick}
    >
      {children}
    </div>
  );
}

type MainPageLinkProps = {
  page: MainPage;
  children: ReactNode;
};

function MainPageLink({ page, children }: MainPageLinkProps): JSX.Element {
  const [mainPage, setMainPage] = useAtom(mainPageAtom);
  return (
    <PageLink active={mainPage == page} onClick={() => setMainPage(page)}>
      {children}
    </PageLink>
  );
}

function MainMenu(): JSX.Element {
  const [showSimPanel, setShowSimPanel] = useAtom(showSimPanelAtom);

  return (
    <nav className="flex space-x-2">
      <MainPageLink page={"trips"}>Züge</MainPageLink>
      <MainPageLink page={"groups"}>Reisende</MainPageLink>
      <MainPageLink page={"stats"}>Statistiken</MainPageLink>
      <PageLink
        active={showSimPanel}
        onClick={() => setShowSimPanel((v) => !v)}
      >
        Was-wäre-wenn-Simulation
      </PageLink>
    </nav>
  );
}

export default MainMenu;
