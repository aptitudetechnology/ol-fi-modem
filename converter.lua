-- Ol-Fi Modem Protocol to FASTA Converter
-- Extracts protocol elements from ol-fi-modem.lua and RFC spec, outputs FASTA file

local modem_file = "ol-fi-modem.lua"
local rfc_file = "ol-fi-modem-draft.rfc-spec.md"
local output_fasta = "ol-fi-modem-converted.fasta"

-- Utility: Read file contents
local function read_file(path)
    local f = io.open(path, "r")
    if not f then return nil end
    local content = f:read("*a")
    f:close()
    return content
end

-- Utility: Write FASTA entry
local function write_fasta_entry(f, name, desc, seq)
    f:write(string.format(">" .. name .. "|%s\n%s\n", desc, seq))
end

-- Extract MVOC types from Lua code
local function extract_mvocs(lua_code)
    local mvocs = {}
    for mvoc in lua_code:gmatch('%["([%w_]+)"%] = {') do
        table.insert(mvocs, mvoc)
    end
    -- Also add hardcoded types from get_mvoc_index
    for mvoc in lua_code:gmatch('%["([%w_]+)"%] = %d+') do
        if not mvocs[mvoc] then table.insert(mvocs, mvoc) end
    end
    return mvocs
end

-- Extract protocol layers and commands from RFC
local function extract_protocol_elements(rfc_text)
    local layers = {}
    for layer in rfc_text:gmatch("Layer: ([%w%s]+)") do
        table.insert(layers, layer)
    end
    local commands = {}
    for cmd in rfc_text:gmatch("%s+([A-Z_]+):") do
        table.insert(commands, cmd)
    end
    return layers, commands
end

-- Main conversion logic
local function main()
    local lua_code = read_file(modem_file) or ""
    local rfc_text = read_file(rfc_file) or ""
    local fasta = io.open(output_fasta, "w")
    if not fasta then
        print("Error: Cannot write to " .. output_fasta)
        return
    end

    -- Extract elements
    local mvocs = extract_mvocs(lua_code)
    local layers, commands = extract_protocol_elements(rfc_text)

    -- Write MVOC entries
    for _, mvoc in ipairs(mvocs) do
        local desc = "type=MVOC|source=Lua|desc=Microbial Volatile Organic Compound"
        local seq = "ATG" .. string.rep("A", 60) -- Placeholder sequence
        write_fasta_entry(fasta, mvoc, desc, seq)
    end

    -- Write protocol layer entries
    for _, layer in ipairs(layers) do
        local desc = "type=Layer|source=RFC|desc=" .. layer
        local seq = "ATG" .. string.rep("C", 40)
        write_fasta_entry(fasta, layer:gsub("%s", "_"), desc, seq)
    end

    -- Write command entries
    for _, cmd in ipairs(commands) do
        local desc = "type=Command|source=RFC|desc=Protocol Command"
        local seq = "ATG" .. string.rep("G", 30)
        write_fasta_entry(fasta, cmd, desc, seq)
    end

    -- Example: Add a frame structure entry
    write_fasta_entry(fasta, "OlFi_Frame", "type=Frame|desc=Protocol Frame Structure", "ATG" .. string.rep("T", 50))

    fasta:close()
    print("Conversion complete. Output: " .. output_fasta)
end

if ... == nil then
    main()
end

