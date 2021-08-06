//
// Created by ascdc on 2021-05-27.
//

#ifndef ENUMV7_INITIAL_SYS_CUH
#define ENUMV7_INITIAL_SYS_CUH

#include "stdIncluded.h"

void initLocale(std::locale);

auto readFile(std::locale, char *) -> std::wstring;


#endif //ENUMV7_INITIAL_SYS_CUH
