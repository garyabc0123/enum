//
// Created by ascdc on 2021-05-27.
//

#include "initial_sys.cuh"

std::wstring readFile(std::locale loc, char *filepath) {
    std::wfstream wfs;
    wfs.imbue(loc);
    wfs.open(filepath, std::fstream::in | std::fstream::binary);

    wfs.seekg(0, std::ios::end);
    size_t fileSize = wfs.tellg();
    //std::experimental::filesystem::path path(filepath);

    //auto fileSize = std::experimental::filesystem::file_size(path);

    std::wstring orgStr(fileSize, ' ');
    wfs.seekg(0);
    wfs.read(&orgStr[0], fileSize);

    wfs.close();
    return orgStr;
}

void initLocale(std::locale utf8) {
    setlocale(LC_CTYPE, "");
    std::wcin.imbue(utf8);
    std::wcout.imbue(utf8);
    std::wcout << L"";
}